from __future__ import annotations
import argparse
import json
import math
import os
import statistics
import time

import numpy as np
import torch

from fanoos import build_prompt, decode_tokens, load_tokenizer
from fanoos import FanoosConfig, FanoosLLM, clean_state_dict
from fanoos import DEFAULT_SYSTEM_PROMPT

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CODE_PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Implement a debounce decorator in Python.",
    "Fix this bug: for i in range(len(items)): if items[i] == None: items.remove(items[i])",
    "Write a Python class for a simple stack with push, pop, and peek methods.",
    "Explain what a Python generator is and give a short example.",
]
CHAT_PROMPTS = [
    "Explain the difference between TCP and UDP like I'm five.",
    "Give me a 3-step plan to learn machine learning.",
    "What are the pros and cons of nuclear energy?",
    "Tell me a short story about a robot that learns to paint.",
    "What is the capital of France and why is it famous?",
]

FACTUAL_PROMPTS = [
    ("What is 2 + 2?", "4"),
    ("What is the capital of Japan?", "tokyo"),
    ("Who wrote Romeo and Juliet?", "shakespeare"),
    ("What color is the sky on a clear day?", "blue"),
    ("What is H2O commonly known as?", "water"),
    ("How many days are in a week?", ("seven", "7")),
    ("What planet is known as the Red Planet?", "mars"),
]

def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["preset"] = None
    cfg = FanoosConfig(**cfg_dict)
    model = FanoosLLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.eval()
    return model, cfg, ckpt

@torch.no_grad()
def eval_perplexity(model, cfg, device, batch_size=8, max_blocks=None):
    val_path  = os.path.join(PROJECT_ROOT, cfg.data_dir, "val.bin")
    mask_path = os.path.join(PROJECT_ROOT, cfg.data_dir, "val_mask.bin")
    if not os.path.isfile(val_path):
        return None, None, 0

    data = np.memmap(val_path, dtype=np.uint32, mode="r")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r") if os.path.isfile(mask_path) else None

    n = (len(data) - 1) // cfg.block_size
    if max_blocks is not None:
        n = min(n, max_blocks)
    if n == 0:
        return None, None, 0

    total_loss, count = 0.0, 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs, ys = [], []
        for i in range(start, end):
            chunk = data[i * cfg.block_size : (i + 1) * cfg.block_size + 1]
            xs.append(chunk[:-1])
            y = chunk[1:].astype(np.int64)
            if mask is not None:
                m = mask[i * cfg.block_size + 1 : (i + 1) * cfg.block_size + 1]
                y[m == 0] = -1
            ys.append(y)

        x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
            _, loss, _ = model(x, y)
        if loss is None:
            continue
        total_loss += loss.item() * len(xs)
        count += len(xs)

    if count == 0:
        return None, None, 0
    mean = total_loss / count
    return mean, math.exp(min(mean, 20)), count

@torch.no_grad()
def bench_generation_speed(model, device, prompt_len=32, max_new_tokens=128,
                           warmup=2, runs=3, eos_token_id=None):
    for _ in range(warmup):
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        model.generate(ids, max_new_tokens=8, temperature=0.8,
                       top_k=50, top_p=0.95, eos_token_id=eos_token_id)

    times = []
    for _ in range(runs):
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.8,
                       top_k=50, top_p=0.95, eos_token_id=eos_token_id)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    tps = [max_new_tokens / t for t in times]
    return {
        "mean_tok_s": statistics.mean(tps),
        "std_tok_s":  statistics.stdev(tps) if len(tps) > 1 else 0.0,
        "min_tok_s":  min(tps),
        "max_tok_s":  max(tps),
        "runs":       len(times),
        "max_new_tokens": max_new_tokens,
    }

@torch.no_grad()
def bench_prefill_speed(model, device, prompt_len=512, warmup=2, runs=3):
    for _ in range(warmup):
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        model(ids, use_cache=True)

    times = []
    for _ in range(runs):
        ids = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(ids, use_cache=True)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_tok_s": prompt_len / statistics.mean(times),
        "prompt_len": prompt_len,
    }

@torch.no_grad()
def qualitative(model, tok, special, device, max_new_tokens=150,
                temperature=0.8, top_k=50, top_p=0.95, show=True):
    results = []
    for kind, prompts in (("code", CODE_PROMPTS), ("chat", CHAT_PROMPTS)):
        for p in prompts:
            text = build_prompt(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
            )
            ids  = tok.encode(text, add_special_tokens=False)
            x    = torch.tensor([ids], device=device)
            out  = model.generate(x, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k, top_p=top_p,
                                  eos_token_id=special["im_end"])
            reply = decode_tokens(tok, out[0, len(ids):].tolist())
            results.append({"category": kind, "prompt": p, "response": reply})
            if show:
                print(f"\n[{kind}] {p}\n  -> {reply}")
    return results

@torch.no_grad()
def factual_accuracy(model, tok, special, device, max_new_tokens=32):
    rows, correct = [], 0
    for q, expected in FACTUAL_PROMPTS:
        text = build_prompt([{"role": "user", "content": q}], add_generation_prompt=True, system_prompt=DEFAULT_SYSTEM_PROMPT)
        ids  = tok.encode(text, add_special_tokens=False)
        x    = torch.tensor([ids], device=device)
        out  = model.generate(x, max_new_tokens=max_new_tokens,
                              temperature=0.0, top_k=1, top_p=1.0,
                              eos_token_id=special["im_end"])
        reply = decode_tokens(tok, out[0, len(ids):].tolist()).lower()
        hit = any(e in reply for e in (expected if isinstance(expected, tuple) else (expected,)))
        correct += int(hit)
        rows.append({"q": q, "expected": expected, "response": reply, "correct": hit})
    total = len(FACTUAL_PROMPTS)
    return {"accuracy": correct / total, "correct": correct, "total": total, "rows": rows}

def report_memory(device):
    if device != "cuda":
        return {}
    return {
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "max_reserved_mb":  torch.cuda.max_memory_reserved()  / (1024 ** 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--val-batches", type=int, default=None,
                    help="cap number of val blocks (default: all)")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-qualitative", action="store_true")
    ap.add_argument("--json-out", default=None, help="write results to this JSON file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.checkpoint
    if ckpt is None:
        for name in ("best.pt", "final.pt", "ckpt.pt"):
            p = os.path.join(PROJECT_ROOT, "checkpoints", name)
            if os.path.isfile(p):
                ckpt = p
                break
    if ckpt is None or not os.path.isfile(ckpt):
        raise SystemExit("No checkpoint found. Train first or pass --checkpoint.")

    tok, special = load_tokenizer()
    model, cfg, raw = load_model(ckpt, device)
    n_params = model.num_parameters()

    results = {
        "checkpoint": ckpt,
        "device": device,
        "params_M": n_params / 1e6,
        "config": {
            "n_layer": cfg.n_layer, "n_head": cfg.n_head, "n_kv_head": cfg.n_kv_head,
            "n_embd": cfg.n_embd, "ffn_dim": cfg.ffn_dim,
            "block_size": cfg.block_size, "vocab_size": cfg.vocab_size,
        },
    }

    print("=" * 62)
    print(f"checkpoint: {ckpt}")
    print(f"device:     {device}")
    print(f"params:     {n_params / 1e6:.2f}M")
    print(f"config:     {cfg.n_layer}L / {cfg.n_head}H / {cfg.n_embd}D / vocab={cfg.vocab_size}")
    print(f"iter:       {raw.get('iter', '?')} | best_val: {raw.get('best_val', '?')}")
    print("=" * 62)

    val_path = os.path.join(PROJECT_ROOT, cfg.data_dir, "val.bin")
    if os.path.isfile(val_path):
        mean, ppl, nblocks = eval_perplexity(model, cfg, device,
                                             max_blocks=args.val_batches)
        if mean is not None:
            print(f"\n[perplexity] val loss {mean:.4f} | ppl {ppl:.2f} (over {nblocks} blocks)")
            results["perplexity"] = {"loss": mean, "ppl": ppl, "blocks": nblocks}
        else:
            print("\n[perplexity] skipped (empty val set)")
    else:
        print("\n[perplexity] skipped (no val.bin)")

    if not args.skip_gen:
        gen = bench_generation_speed(model, device, max_new_tokens=args.max_new_tokens)
        print(f"\n[gen speed] {gen['mean_tok_s']:.1f} ± {gen['std_tok_s']:.1f} tok/s "
              f"(min {gen['min_tok_s']:.1f}, max {gen['max_tok_s']:.1f}, n={gen['runs']})")
        results["generation_speed"] = gen

        prefill_len = min(512, cfg.block_size)
        pre = bench_prefill_speed(model, device, prompt_len=prefill_len)
        print(f"[prefill]   {pre['mean_tok_s']:.1f} tok/s at len {pre['prompt_len']}")
        results["prefill_speed"] = pre

    if not args.skip_qualitative:
        print("\n" + "=" * 62)
        print("QUALITATIVE SAMPLES")
        print("=" * 62)
        results["qualitative"] = qualitative(
            model, tok, special, device, max_new_tokens=args.max_new_tokens, show=True)

        print("\n" + "=" * 62)
        print("FACTUAL QA")
        print("=" * 62)
        fa = factual_accuracy(model, tok, special, device)
        for r in fa["rows"]:
            status = "OK  " if r["correct"] else "MISS"
            print(f"[{status}] {r['q']:45s} -> {r['response'][:60]}")
        print(f"accuracy: {fa['correct']}/{fa['total']} = {fa['accuracy']:.0%}")
        results["factual"] = fa

    mem = report_memory(device)
    if mem:
        print(f"\n[memory] max allocated {mem['max_allocated_mb']:.1f} MB | "
              f"reserved {mem['max_reserved_mb']:.1f} MB")
        results["memory"] = mem

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[benchmark] results written to {args.json_out}")

if __name__ == "__main__":
    main()