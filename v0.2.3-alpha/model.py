import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# ─────────────────────────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


# ─────────────────────────────────────────────────────────────
#  Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────────────────────
def precompute_freqs_cis(head_dim: int, max_seq_len: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Return complex64 rotation frequencies of shape (max_seq_len, head_dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        fc = freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(x_c * fc).flatten(3).type_as(x)

    return _rotate(q), _rotate(k)


# ─────────────────────────────────────────────────────────────
#  Causal Self-Attention (with RoPE + Flash Attention)
# ─────────────────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        self.c_attn   = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj   = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Flash Attention unavailable – using slow manual attention.")
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


# ─────────────────────────────────────────────────────────────
#  SwiGLU Feed-Forward (replaces GELU MLP)
# ─────────────────────────────────────────────────────────────
def _find_multiple(n: int, k: int) -> int:
    return n if n % k == 0 else n + k - (n % k)


class SwiGLU(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        hidden = _find_multiple(int(2 * 4 * config.n_embd / 3), 64)
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─────────────────────────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────────────────────────
class Block(nn.Module):

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp  = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x


# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    block_size:  int   = 512
    vocab_size:  int   = 50257
    n_layer:     int   = 8
    n_head:      int   = 8
    n_embd:      int   = 512
    dropout:     float = 0.0
    rope_theta:  float = 10000.0
    bias:        bool  = False


# ─────────────────────────────────────────────────────────────
#  FanoosLLM (GPT class)
# ─────────────────────────────────────────────────────────────
class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        head_dim = config.n_embd // config.n_head
        freqs = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("c_proj.weight", "down_proj.weight")):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        total = self.get_num_params()
        print(f"FanoosLLM ready · {total / 1e6:.2f} M parameters")

    # ------------------------------------------------------------------
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} > block_size {self.config.block_size}"
        )

        x = self.transformer.drop(self.transformer.wte(idx))
        freqs = self.freqs_cis[:t]

        for block in self.transformer.h:
            x = block(x, freqs)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ------------------------------------------------------------------
    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        head_dim = self.config.n_embd // self.config.n_head
        self.freqs_cis = precompute_freqs_cis(
            head_dim, block_size, self.config.rope_theta
        )

    # ------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay = [p for n, p in param_dict.items() if p.dim() < 2]
        groups  = [
            {"params": decay,   "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        print(
            f"Decay: {sum(p.numel() for p in decay):,} params | "
            f"No-decay: {sum(p.numel() for p in nodecay):,} params"
        )
        use_fused = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        optimizer = torch.optim.AdamW(
            groups, lr=learning_rate, betas=betas,
            **({"fused": True} if use_fused else {}),
        )
        print(f"Fused AdamW: {use_fused}")
        return optimizer

    # ------------------------------------------------------------------
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N   = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops = (6 * N + 12 * L * H * Q * T) * T * fwdbwd_per_iter
        return flops / (dt * 312e12)                       # vs A100 peak

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)
        return idx