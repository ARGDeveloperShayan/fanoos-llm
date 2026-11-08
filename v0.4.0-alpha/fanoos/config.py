from __future__ import annotations
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

PRESETS: dict[str, dict[str, Any]] = {
    "fanoos-15m": {
        "n_layer": 8,
        "n_head": 6, "n_kv_head": 2,
        "n_embd": 384, "ffn_dim": 1024,
        "block_size": 512, "sliding_window_size": 512,
        "dropout": 0.0,
    },
    "fanoos-40m": {
        "n_layer": 10,
        "n_head": 8, "n_kv_head": 2,
        "n_embd": 512, "ffn_dim": 1408,
        "block_size": 512, "sliding_window_size": 512,
        "dropout": 0.0,
    },
    "fanoos-130m": {
        "n_layer": 12,
        "n_head": 8, "n_kv_head": 2,
        "n_embd": 768, "ffn_dim": 2048,
        "block_size": 512, "sliding_window_size": 512,
        "dropout": 0.05,
    },
    "fanoos-400m": {
        "n_layer": 20,
        "n_head": 16, "n_kv_head": 4,
        "n_embd": 1024, "ffn_dim": 2816,
        "block_size": 1024, "sliding_window_size": 1024,
        "dropout": 0.1,
    },
    "fanoos-800m": {
        "n_layer": 24,
        "n_head": 16, "n_kv_head": 4,
        "n_embd": 1280, "ffn_dim": 4096,
        "block_size": 1024, "sliding_window_size": 1024,
        "dropout": 0.1,
    },
    "fanoos-1b": {
        "n_layer": 28,
        "n_head": 16, "n_kv_head": 4,
        "n_embd": 1792, "ffn_dim": 4864,
        "block_size": 1024, "sliding_window_size": 1024,
        "dropout": 0.1,
    },
}

@dataclass
class FanoosConfig:
    vocab_size: Optional[int] = None
    vocab_size_padded_multiple: int = 128

    # ---------------------------- Architecture ---------------------------
    block_size: int = 1024
    sliding_window_size: int = 1024
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    ffn_dim: int = 1408
    rope_theta: float = 10000.0
    dropout: float = 0.1
    bias: bool = False
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True

    # ---------------------------- Generation -----------------------------
    temperature: float = 0.6
    top_k: int = 50
    top_p: float = 0.95

    # ----------------------------- Training ------------------------------
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_train_iters: int = 20000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # ------------------------------ Paths --------------------------------
    tokenizer_dir: str = "tokenizer"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    export_dir: str = "export"

    preset: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.preset:
            preset = PRESETS.get(self.preset)
            if preset is None:
                raise ValueError(f"Unknown preset '{self.preset}'. Options: {list(PRESETS)}")
            for key, value in preset.items():
                setattr(self, key, value)
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_kv_head ({self.n_kv_head}) must divide n_head ({self.n_head})")
        head_dim = self.n_embd // self.n_head
        if self.n_embd % self.n_head != 0 or head_dim < 32:
            raise ValueError("n_embd must divide n_head and give head_dim >= 32")

    # ------------------------------ Helpers ------------------------------
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_params_estimate(self) -> int:
        head_dim = self.head_dim
        per_layer = (
            self.n_embd * self.n_head * head_dim
            + 2 * self.n_embd * self.n_kv_head * head_dim
            + self.n_head * head_dim * self.n_embd
            + 3 * self.n_embd * self.ffn_dim
            + 2 * self.n_embd
        )
        return per_layer * self.n_layer + self.vocab_size_resolved * self.n_embd

    def resolve_vocab_size(self, tokenizer_vocab: int) -> int:
        if self.vocab_size is None:
            multiple = self.vocab_size_padded_multiple
            self.vocab_size = math.ceil(tokenizer_vocab / multiple) * multiple
        if self.vocab_size < tokenizer_vocab:
            raise ValueError(
                f"vocab_size={self.vocab_size} is smaller than the tokenizer "
                f"vocabulary ({tokenizer_vocab}). Token ids would go out of range."
            )
        return self.vocab_size

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preset"] = None
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FanoosConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:
        lines = ["FanoosConfig("]
        for f in fields(self):
            if f.name == "preset":
                continue
            lines.append(f"  {f.name}={getattr(self, f.name)!r}")
        lines.append(")")
        return "\n".join(lines)

if __name__ == "__main__":
    cfg = FanoosConfig(preset="fanoos-400m", vocab_size=154880)
    print(repr(cfg))
    print(f"Estimated parameters: {cfg.n_params_estimate / 1e6:.1f}M")