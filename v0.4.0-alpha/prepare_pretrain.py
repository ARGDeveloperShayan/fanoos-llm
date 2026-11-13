from __future__ import annotations
import os as _os, sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = str(_Path(__file__).resolve().parent)
_sys.path.insert(0, _PROJECT_ROOT)

import argparse, glob, os
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from fanoos import load_tokenizer

DATA_DIR  = os.path.join(_PROJECT_ROOT, "assets", "datasets")
CACHE_DIR = os.path.join(_PROJECT_ROOT, "assets", "pretrain_bins")
OUT_DIR   = os.path.join(_PROJECT_ROOT, "data_pretrain")

SOURCES = [
    ("fineweb", "fineweb-edu-1B-*.parquet",        "text", 0.45),
    ("wiki",    "wikipedia-20231101-en-*.parquet", "text", 0.15),
    ("cosmo",   "cosmopedia-v2-*.parquet",         "text", 0.25),
    ("python",  "python-code-*.parquet",           "text", 0.10),
]

def find_files(pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, pattern)))

def detect_text_column(path: str, preferred: str) -> str | None:
    names = pq.ParquetFile(path).schema_arrow.names
    if preferred in names:
        return preferred
    for cand in ("text", "content", "raw_content", "markdown"):
        if cand in names:
            return cand
    return None

def tokenize_source(name: str, files: list[str], text_col: str, eot_id: int,
                    tok, cache_bin: str, cache_mask: str) -> int:
    if (os.path.isfile(cache_bin) and os.path.isfile(cache_mask)
            and os.path.getsize(cache_bin) > 0
            and os.path.getsize(cache_bin) == 4 * os.path.getsize(cache_mask)):
        n = os.path.getsize(cache_bin) // 4
        print(f"[prep] {name}: cached ({n/1e9:.2f}B tokens)")
        return n

    print(f"[prep] {name}: tokenizing {len(files)} file(s)")
    tmp_bin, tmp_mask = cache_bin + ".tmp", cache_mask + ".tmp"
    n_tokens = 0
    buf_ids, buf_mask = [], []
    BUF_TARGET = 200_000

    def flush(fbin, fmask):
        nonlocal buf_ids, buf_mask
        if not buf_ids:
            return
        np.asarray(buf_ids, dtype=np.uint32).tofile(fbin)
        np.asarray(buf_mask, dtype=np.uint8).tofile(fmask)
        buf_ids, buf_mask = [], []

    with open(tmp_bin, "wb") as fbin, open(tmp_mask, "wb") as fmask:
        for fi, path in enumerate(files):
            pf = pq.ParquetFile(path)
            with tqdm(total=pf.metadata.num_rows,
                      desc=f"{name} [{fi+1}/{len(files)}]", unit="docs",
                      leave=False) as bar:
                for batch in pf.iter_batches(batch_size=500):
                    texts = [t for t in batch.column(text_col).to_pylist() if t]
                    if not texts:
                        bar.update(500)
                        continue
                    encoded = tok(texts, add_special_tokens=False,
                                  truncation=True, max_length=4096)["input_ids"]
                    for ids in encoded:
                        ids.append(eot_id)
                        buf_ids.extend(ids)
                        buf_mask.extend([1] * len(ids))
                        n_tokens += len(ids)
                        if len(buf_ids) >= BUF_TARGET:
                            flush(fbin, fmask)
                    bar.update(500)
        flush(fbin, fmask)

    os.replace(tmp_bin, cache_bin)
    os.replace(tmp_mask, cache_mask)
    print(f"[prep] {name}: {n_tokens/1e9:.2f}B tokens")
    return n_tokens

def merge_sources(sources, block_size: int, val_fraction: float,
                  target_tokens: int | None):
    items = []
    for s in sources:
        n = os.path.getsize(s["bin"]) // 4
        items.append({**s, "len": n})

    available = sum(s["len"] for s in items)
    if target_tokens is None or target_tokens > available:
        target_tokens = available
    print(f"[prep] merge: available {available/1e9:.2f}B, target {target_tokens/1e9:.2f}B")

    wsum = sum(s["weight"] for s in items)
    for s in items:
        s["take"] = min(int(target_tokens * s["weight"] / wsum), s["len"])

    CHUNK = 1_000_000
    chunks = []
    for si, s in enumerate(items):
        remaining = s["take"]
        pos = 0
        while remaining > 0:
            sz = min(CHUNK, remaining)
            chunks.append((si, pos, sz))
            pos += sz
            remaining -= sz

    rng = np.random.default_rng(1234)
    rng.shuffle(chunks)

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = {k: os.path.join(OUT_DIR, f"{k}.bin") for k in
             ("train", "train_mask", "val", "val_mask")}
    fb_tr = open(paths["train"], "wb");      fm_tr = open(paths["train_mask"], "wb")
    fb_va = open(paths["val"], "wb");        fm_va = open(paths["val_mask"], "wb")

    src_data = [np.memmap(s["bin"],  dtype=np.uint32, mode="r") for s in items]
    src_mask = [np.memmap(s["mask"], dtype=np.uint8,  mode="r") for s in items]

    n_tr = n_va = 0
    for ci, (si, off, sz) in enumerate(chunks):
        d, m = src_data[si][off:off+sz], src_mask[si][off:off+sz]
        if ci % 400 == 0:
            d.tofile(fb_va); m.tofile(fm_va); n_va += sz
        else:
            d.tofile(fb_tr); m.tofile(fm_tr); n_tr += sz
        if ci % 50 == 0:
            print(f"[prep] merge: {ci+1}/{len(chunks)} chunks, "
                  f"{(n_tr+n_va)/1e9:.2f}B tokens", flush=True)

    for f in (fb_tr, fm_tr, fb_va, fm_va):
        f.close()

    for key, dtype in (("train", np.uint32), ("val", np.uint32),
                       ("train_mask", np.uint8), ("val_mask", np.uint8)):
        p = paths[key]
        n = os.path.getsize(p) // np.dtype(dtype).itemsize
        with open(p, "r+b") as f:
            f.truncate((n // block_size) * block_size * np.dtype(dtype).itemsize)

    print(f"[prep] done: train {n_tr/1e9:.2f}B | val {n_va/1e6:.0f}M tokens")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=512)
    args = ap.parse_args()

    tok, _ = load_tokenizer()
    eot_id = tok.convert_tokens_to_ids("<|endoftext|>")

    os.makedirs(CACHE_DIR, exist_ok=True)
    sources = []
    for name, pattern, pref_col, weight in SOURCES:
        files = find_files(pattern)
        if not files:
            print(f"[prep] skip {name}: no files matching {pattern}")
            continue
        col = detect_text_column(files[0], pref_col)
        if col is None:
            print(f"[prep] skip {name}: no text column in {os.path.basename(files[0])}")
            continue
        cache_bin  = os.path.join(CACHE_DIR, f"{name}.bin")
        cache_mask = os.path.join(CACHE_DIR, f"{name}_mask.bin")
        tokenize_source(name, files, col, eot_id, tok, cache_bin, cache_mask)
        sources.append({"name": name, "bin": cache_bin, "mask": cache_mask,
                        "weight": weight})

    if not sources:
        raise SystemExit("[prep] no usable sources — need parquet files with a 'text' column in assets/datasets/")

    merge_sources(sources, args.block_size, 0.002, args.target_tokens)

if __name__ == "__main__":
    main()