from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import FanoosConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype)).to(dtype)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, block_size: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build(block_size)

    def _build(self, max_len: int):
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor):
        need = int(positions.max().item()) + 1
        if need > self.cos_cached.shape[0]:
            self._build(need)
        return self.cos_cached[positions], self.sin_cached[positions]

    def get_range(self, start: int, length: int, device):
        if start + length > self.cos_cached.shape[0]:
            self._build(start + length)
        return (self.cos_cached[start:start+length].to(device),
                self.sin_cached[start:start+length].to(device))

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin

class KVCache:
    def __init__(self, n_layers: int):
        self.keys: list[Optional[torch.Tensor]] = [None] * n_layers
        self.values: list[Optional[torch.Tensor]] = [None] * n_layers
        self.pos = 0

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, crop_to: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer_idx] is None:
            self.keys[layer_idx], self.values[layer_idx] = k, v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], k], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=2)
        if crop_to is not None and self.keys[layer_idx].shape[2] > crop_to:
            self.keys[layer_idx] = self.keys[layer_idx][:, :, -crop_to:, :]
            self.values[layer_idx] = self.values[layer_idx][:, :, -crop_to:, :]
        return self.keys[layer_idx], self.values[layer_idx]

    def length(self) -> int:
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def key_length(self, layer_idx: int) -> int:
        k = self.keys[layer_idx]
        return 0 if k is None else k.shape[2]

    def to_tuple(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((k, v) for k, v in zip(self.keys, self.values) if k is not None)

    @classmethod
    def from_tuple(cls, past) -> "KVCache":
        cache = cls(len(past))
        for i, (k, v) in enumerate(past):
            cache.keys[i], cache.values[i] = k, v
        cache.pos = past[0][0].shape[2]
        return cache

class Attention(nn.Module):
    def __init__(self, cfg: FanoosConfig, layer_idx: int, rope: RotaryEmbedding):
        super().__init__()
        assert cfg.n_head % cfg.n_kv_head == 0
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.window = cfg.sliding_window_size
        self.attn_dropout = cfg.dropout
        self.rope = rope

        hidden_q = cfg.n_head * self.head_dim
        hidden_kv = cfg.n_kv_head * self.head_dim
        self.q_proj = nn.Linear(cfg.n_embd, hidden_q, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, hidden_kv, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, hidden_kv, bias=cfg.bias)
        self.o_proj = nn.Linear(hidden_q, cfg.n_embd, bias=cfg.bias)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        b, kv, t, hd = x.shape
        return x[:, :, None, :, :].expand(b, kv, n_rep, t, hd).reshape(b, kv * n_rep, t, hd)

    def _sliding_window_mask(
        self, q_len: int, kv_len: int, past_len: int, k_offset: int, device, dtype
    ) -> torch.Tensor:
        q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
        k_pos = torch.arange(kv_len, device=device)[None, :] + k_offset
        allowed = (k_pos <= q_pos) & (k_pos > q_pos - self.window)
        return torch.zeros(q_len, kv_len, device=device, dtype=dtype).masked_fill(
            ~allowed, float("-inf")
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
        start_pos: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        physical = 0 if cache is None else cache.key_length(self.layer_idx)
        if start_pos is not None:
            past_len = start_pos
        elif cache is None:
            past_len = 0
        else:
            past_len = physical

        positions = torch.arange(past_len, past_len + T, device=x.device)
        cos, sin = self.rope(positions)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        decoding = T == 1 and cache is not None
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v, crop_to=self.window if decoding else None)
        kv_len = k.shape[2]
        k_offset = (past_len + T) - kv_len

        if decoding:
            attn_mask, is_causal = None, False
        elif kv_len == T and T <= self.window:
            attn_mask, is_causal = None, True
        else:
            attn_mask = self._sliding_window_mask(T, kv_len, past_len, k_offset, x.device, q.dtype)
            is_causal = False

        n_rep = self.n_head // self.n_kv_head
        y = F.scaled_dot_product_attention(
            q, self._repeat_kv(k, n_rep), self._repeat_kv(v, n_rep),
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: FanoosConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, cfg: FanoosConfig, layer_idx: int, rope: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.attn = Attention(cfg, layer_idx, rope)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None, start_pos: Optional[int] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), cache, start_pos))
        x = x + self.dropout(self.mlp(self.ffn_norm(x)))
        return x

class FanoosLLM(nn.Module):
    def __init__(self, cfg: FanoosConfig):
        super().__init__()
        self.cfg = cfg
        cfg.resolve_vocab_size(cfg.vocab_size or 154880)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList(Block(cfg, i, RotaryEmbedding(cfg.head_dim, cfg.block_size, cfg.rope_theta))
                           for i in range(cfg.n_layer)),
            norm_f=RMSNorm(cfg.n_embd, cfg.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.transformer.wte.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[KVCache]]:
        if isinstance(past_key_values, tuple):
            past_key_values = KVCache.from_tuple(past_key_values)
        cache = past_key_values if (use_cache or past_key_values is not None) else None
        if use_cache and cache is None:
            cache = KVCache(self.cfg.n_layer)
        start_pos = cache.pos if cache is not None else 0

        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        use_ckpt = (
            self.gradient_checkpointing and self.training and cache is None and torch.is_grad_enabled()
        )
        for block in self.transformer.h:
            if use_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False, preserve_rng_state=True
                )
            else:
                x = block(x, cache, start_pos)
        if cache is not None:
            cache.pos = start_pos + idx.shape[1]

        x = self.transformer.norm_f(x)
        if targets is not None:
            loss = self._chunked_cross_entropy(x, targets)
            logits = None
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss, (cache if use_cache else None)

    def _chunked_cross_entropy(self, x: torch.Tensor, targets: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        def _loss(xi, ti):
            logits_i = self.lm_head(xi)
            return F.cross_entropy(
                logits_i.reshape(-1, logits_i.size(-1)),
                ti.reshape(-1), ignore_index=-1, reduction="sum",
            )

        total = x.new_zeros((), dtype=torch.float32)
        count = 0
        use_ckpt = torch.is_grad_enabled() and self.training and self.gradient_checkpointing
        for xi, ti in zip(x.split(chunk, dim=1), targets.split(chunk, dim=1)):
            if use_ckpt:
                loss_i = torch.utils.checkpoint.checkpoint(
                    _loss, xi, ti, use_reentrant=False, preserve_rng_state=False
                )
            else:
                loss_i = _loss(xi, ti)
            total = total + loss_i
            count = count + (ti != -1).sum()
        return total / count.clamp(min=1)

    @torch.no_grad()
    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative - probs > top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.gather(-1, sorted_idx.argsort(-1))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> torch.Tensor:
        self.eval()
        device = idx.device
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device)

        logits, _, cache = self(idx, past_key_values=past_key_values, use_cache=True)
        for _ in range(max_new_tokens):
            next_tok = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            next_tok = next_tok.masked_fill(finished[:, None], self.cfg.vocab_size - 1)
            if eos_token_id is not None:
                finished |= next_tok.squeeze(1) == eos_token_id
                if bool(finished.all()):
                    idx = torch.cat([idx, next_tok], dim=1)
                    break
            idx = torch.cat([idx, next_tok], dim=1)
            logits, _, cache = self(next_tok, past_key_values=cache, use_cache=True)
        return idx

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_word_embeddings is False:
            n -= self.transformer.wte.weight.numel()
        return n

    def configure_optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim < 2 or "norm" in name.lower()) else decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return groups, betas, lr

def build_model(cfg: Optional[FanoosConfig] = None, tokenizer_vocab: Optional[int] = None,
                device: str = "cpu") -> FanoosLLM:
    cfg = cfg or FanoosConfig()
    if tokenizer_vocab is not None:
        cfg.resolve_vocab_size(tokenizer_vocab)
    model = FanoosLLM(cfg)
    return model.to(device)

if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = FanoosConfig(preset="fanoos-400m", vocab_size=154880, block_size=512, sliding_window_size=128)
    model = FanoosLLM(cfg)
    print(f"parameters: {model.num_parameters() / 1e6:.1f}M")

    idx = torch.randint(0, cfg.vocab_size, (2, 100))
    targets = torch.randint(0, cfg.vocab_size, (2, 100))
    _, loss, _ = model(idx, targets)
    print("full forward: loss:", loss.item())

    cfg.dropout = 0.0
    model.eval()
    with torch.no_grad():
        full_logits, _, _ = model(idx, use_cache=False)
        logits, _, cache = model(idx[:, :50], use_cache=True)
        step_logits, _, cache = model(idx[:, 50:], past_key_values=cache, use_cache=True)
        max_diff = (step_logits[:, -1] - full_logits[:, -1]).abs().max().item()
    print(f"cache consistency (max |diff|): {max_diff:.2e}")

    out = model.generate(idx[:, :10], max_new_tokens=20, temperature=1.0, eos_token_id=None)
    print("generate:", out.shape)