from __future__ import annotations
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.backends.cuda as cuda
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from fanoos import FanoosConfig
from fanoos import load_tokenizer
from fanoos import FanoosLLM
from fanoos import clean_state_dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_bin(path, dtype, label):
    if not os.path.isfile(path):
        raise SystemExit(
            f"[train] missing {label}: {path}\n"
            f"        Run prepare_pretrain.py (pretrain) or prepare.py (SFT) first."
        )
    if os.path.getsize(path) == 0:
        raise SystemExit(
            f"[train] empty {label}: {path}\n"
            f"        The prepare step produced no data."
        )
    return np.memmap(path, dtype=dtype, mode="r")

def get_batch(data, mask, block_size, batch_size, device, aligned=False):
    if aligned:
        ix = np.random.randint(0, (len(data) - 1) // block_size, size=batch_size) * block_size
    else:
        ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x_np = data[ix[:, None] + np.arange(block_size)].astype(np.int64)
    y_np = data[ix[:, None] + np.arange(1, block_size + 1)].astype(np.int64)
    m_np = mask[ix[:, None] + np.arange(1, block_size + 1)]
    y_np[m_np == 0] = -1

    if device == "cuda":
        x = torch.from_numpy(x_np).pin_memory().to(device, non_blocking=True)
        y = torch.from_numpy(y_np).pin_memory().to(device, non_blocking=True)
    else:
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
    return x, y

def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.max_train_iters:
        return cfg.min_learning_rate
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_train_iters - cfg.warmup_iters)
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (
        1.0 + math.cos(math.pi * ratio)
    )

def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        print("[train] optimizer: bitsandbytes AdamW8bit")
    except ImportError:
        opt = torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95),
                                fused=torch.cuda.is_available())
        print("[train] optimizer: AdamW8bit")
        print("[train] bitsandbytes not installed. To install, run: pip install bitsandbytes")
    return opt

def export_hf(model, cfg, out_dir):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    save_file(state, os.path.join(out_dir, "model.safetensors"))
    cfg.save(os.path.join(out_dir, "config.json"))

    tok_src = os.path.join(PROJECT_ROOT, cfg.tokenizer_dir)
    if os.path.isdir(tok_src):
        import shutil
        for fname in os.listdir(tok_src):
            if fname != "chatml_special_tokens.json":
                shutil.copy2(os.path.join(tok_src, fname), os.path.join(out_dir, fname))
    print(f"[train] exported Hugging Face format to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fanoos-15m", choices=list(__import__("fanoos").PRESETS))
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    # ap.add_argument("--eval-interval", type=int, default=250)
    # ap.add_argument("--eval-iters", type=int, default=50)
    # ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true")

    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-iters",    type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=250)

    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--aligned", action="store_true")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[train] WARNING: CUDA not available, training on CPU (very slow).")

    cfg = FanoosConfig(preset=args.preset)
    if args.max_iters:   cfg.max_train_iters = args.max_iters
    if args.micro_batch: cfg.micro_batch_size = args.micro_batch
    if args.grad_accum:  cfg.gradient_accumulation_steps = args.grad_accum
    if args.block_size:  cfg.block_size = args.block_size
    if args.data_dir:  cfg.data_dir = args.data_dir
    if args.ckpt_dir:  cfg.checkpoint_dir = args.ckpt_dir
    if args.lr:        cfg.learning_rate = args.lr
    if args.min_lr:    cfg.min_learning_rate = args.min_lr
    if args.warmup:    cfg.warmup_iters = args.warmup
    if args.dropout is not None: cfg.dropout = args.dropout

    tok, special = load_tokenizer()
    cfg.resolve_vocab_size(len(tok))
    print(repr(cfg))
    print(f"[train] tokenizer vocab {len(tok):,} -> model vocab {cfg.vocab_size:,}")

    data_dir = os.path.join(PROJECT_ROOT, cfg.data_dir)
    train_data = load_bin(os.path.join(data_dir, "train.bin"),      np.uint32, "train.bin")
    val_data   = load_bin(os.path.join(data_dir, "val.bin"),        np.uint32, "val.bin")
    train_mask = load_bin(os.path.join(data_dir, "train_mask.bin"), np.uint8,  "train_mask.bin")
    val_mask   = load_bin(os.path.join(data_dir, "val_mask.bin"),   np.uint8,  "val_mask.bin")
    print(f"[train] train.bin: {len(train_data):,} tokens | val.bin: {len(val_data):,} tokens")

    model = FanoosLLM(cfg).to(device)
    raw_model = model
    print(f"[train] parameters: {model.num_parameters() / 1e6:.2f}M")
    if args.init_from:
        sd = clean_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"])
        res = model.load_state_dict(sd, strict=False)
        assert not res.unexpected_keys and set(res.missing_keys) <= {"lm_head.weight"}, res
        print("[train] init from", args.init_from, "OK")

    if not args.no_ckpt and model.num_parameters() > 100_000_000:
        model.gradient_checkpointing_enable()
        print("[train] gradient checkpointing: ON (large model)")
    else:
        print("[train] gradient checkpointing: OFF")
    if args.compile:
        model = torch.compile(model)
        print("[train] torch.compile: ON")

    optimizer = build_optimizer(model, cfg)

    ckpt_dir = os.path.join(PROJECT_ROOT, cfg.checkpoint_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

    start_iter, best_val = 0, float("inf")
    if args.resume and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        raw_model.load_state_dict(clean_state_dict(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt.get("best_val", float("inf"))
        print(f"[train] resumed from {ckpt_path} at iter {start_iter}")

    n_eval_iters = args.eval_iters

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split, data, msk in (("train", train_data, train_mask),
                                 ("val",   val_data,   val_mask)):
            losses = torch.zeros(n_eval_iters)
            for k in range(n_eval_iters):
                x, y = get_batch(data, msk, cfg.block_size, cfg.micro_batch_size, device, aligned=args.aligned)
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=device == "cuda"):
                    _, loss, _ = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model.train()
    t0 = time.time()
    tokens_seen = 0
    print(f"[train] starting: {cfg.max_train_iters} iters | effective batch "
          f"{cfg.micro_batch_size * cfg.gradient_accumulation_steps * cfg.block_size:,} tokens/step")

    for it in range(start_iter, cfg.max_train_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        step_t0 = time.time()

        lr = get_lr(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro in range(cfg.gradient_accumulation_steps):
            x, y = get_batch(train_data, train_mask, cfg.block_size, cfg.micro_batch_size, device, aligned=args.aligned)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device == "cuda"):
                _, loss, _ = model(x, y)
            (loss / cfg.gradient_accumulation_steps).backward()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        step_dt = time.time() - step_t0

        if it < 20 or it % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"iter {it:6d}/{cfg.max_train_iters} | loss {loss.item():.4f} | "
                f"lr {lr:.2e} | step {step_dt:.2f}s | "
                f"{tokens_seen / elapsed / 1e3:.0f}k tok/s",
                flush=True,
            )

        if (it % args.eval_interval == 0 and it > 0) or it == cfg.max_train_iters - 1:
            losses = estimate_loss()
            print(f"[eval] iter {it}: train {losses['train']:.4f} val {losses['val']:.4f} "
                  f"(ppl {math.exp(min(losses['val'], 20)):.2f})")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": raw_model.state_dict(), "cfg": cfg.to_dict(),
                            "iter": it, "best_val": best_val},
                           os.path.join(ckpt_dir, "best.pt"))

        if (it % args.save_interval == 0 and it > 0) or it == cfg.max_train_iters - 1:
            torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                        "cfg": cfg.to_dict(), "iter": it, "best_val": best_val},
                       ckpt_path)
            print(f"[train] checkpoint saved to {ckpt_path}", flush=True)

    torch.save({"model": raw_model.state_dict(), "cfg": cfg.to_dict(),
                "iter": cfg.max_train_iters},
               os.path.join(ckpt_dir, "final.pt"))
    export_hf(model, cfg, os.path.join(PROJECT_ROOT, cfg.export_dir, "fanoos-ai"))
    print("[train] done.")

if __name__ == "__main__":
    main()