from __future__ import annotations
import argparse
import json
import os
import random
import numpy as np
from fanoos import DEFAULT_SYSTEM_PROMPT, load_tokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "assets", "datasets")

DATASETS = [
    # ---- Conversation ----
    ("everyday-conversations-llama3.1-2k.parquet", "conversation"),

    # ---- Code ----
    ("python-code-instructions-85k.jsonl", "code"),
    ("code_review_improvement.parquet", "code"),
    ("complexity_edge_cases.parquet", "code"),
    ("explain_code_concept.parquet", "code"),
    ("library_api_usage.parquet", "code"),
    ("refactor_discussion.parquet", "code"),

    # ---- Knowledge ----
    ("databricks-dolly-15k.jsonl", "knowledge"),

    # ---- Reasoning ----
    ("General_Inquiry_Thinking-Chain-Of-Thought_6K.json", "reasoning"),
]

CATEGORY_WEIGHTS = {
    "conversation": 0.30,
    "code": 0.30,
    "knowledge": 0.25,
    "reasoning": 0.15,
}

DROP_KEYWORDS = (
    "premise", "hypothesis", "entailed", "entailment",
    "based on the passage", "given the passage", "given the text",
    "what is the main idea", "which option", "select the", "choose the correct",
    "read the following", "answer the question based",
)

BAD_SHORT_ASSISTANT = {
    "yes", "no", "true", "false", "entailed", "not entailed",
    "neutral", "contradiction", "positive", "negative",
    "yes.", "no.", "entailed.", "neutral.", "a", "b", "c", "d",
}

_ROLE_MAP = {
    "system": "system", "user": "user", "assistant": "assistant",
    "human": "user", "gpt": "assistant", "bing": "assistant",
    "bot": "assistant", "ai": "assistant",
}

def _first_user(messages):
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")
    return ""

def _last_assistant(messages):
    last = ""
    for m in messages:
        if m.get("role") == "assistant":
            last = m.get("content", "").strip().lower()
    return last

def is_chat_worthy(messages):
    u = _first_user(messages).lower()
    if any(k in u for k in DROP_KEYWORDS):
        return False
    if _last_assistant(messages) in BAD_SHORT_ASSISTANT:
        return False
    wc = len(u.split())
    if wc < 4 and "?" not in u and u.strip(" .!") not in (
        "hi", "hello", "hey", "thanks", "thank you"
    ):
        return False
    return True

def _coerce_role(r):
    if r is None:
        return None
    return _ROLE_MAP.get(str(r).lower())

def _to_plain_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    try:
        return list(v)
    except TypeError:
        return None

def _from_messages_field(msgs):
    out = []
    for m in msgs or []:
        if not isinstance(m, dict):
            continue
        r = _coerce_role(m.get("role") or m.get("from") or m.get("speaker"))
        c = m.get("content") or m.get("value") or m.get("text") or m.get("message")
        if r and c and isinstance(c, str):
            out.append({"role": r, "content": c.strip()})
    return out or None

def adapt_record(rec):
    if not isinstance(rec, dict):
        return None

    for key in ("messages", "conversation", "conversations", "turns",
                "dialogue", "chat", "conversation_history"):
        v = _to_plain_list(rec.get(key))
        if v and isinstance(v[0], dict):
            msgs = _from_messages_field(v)
            if msgs and len(msgs) >= 2:
                return msgs

    inst = rec.get("instruction") or rec.get("prompt") or rec.get("question") or rec.get("query")
    out = rec.get("output") or rec.get("response") or rec.get("answer") or rec.get("completion") or rec.get("chosen")
    ctx = rec.get("input") or rec.get("context") or rec.get("passage") or ""
    if isinstance(inst, str) and isinstance(out, str) and inst.strip() and out.strip():
        user = f"{inst.strip()}\n\n{ctx.strip()}" if isinstance(ctx, str) and ctx.strip() else inst.strip()
        return [{"role": "user", "content": user},
                {"role": "assistant", "content": out.strip()}]

    q = rec.get("question")
    a = rec.get("answer")
    if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
        return [{"role": "user", "content": q.strip()},
                {"role": "assistant", "content": a.strip()}]

    return None

def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "train", "rows", "records"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data if isinstance(data, list) else []

def _load_parquet(path):
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except Exception:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=path, split="train")
        return [dict(r) for r in ds]

def load_file(path):
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    if path.endswith(".json"):
        return _load_json(path)
    if path.endswith(".parquet"):
        return _load_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def load_all():
    by_category = {c: [] for c in CATEGORY_WEIGHTS}
    for fname, cat in DATASETS:
        path = os.path.join(DATA_ROOT, fname)
        if not os.path.isfile(path):
            print(f"[prepare] MISSING: {path} (skipping)")
            continue
        raw = load_file(path)
        kept = 0
        for rec in raw:
            msgs = adapt_record(rec)
            if msgs and len(msgs) >= 2 and is_chat_worthy(msgs):
                by_category[cat].append(msgs)
                kept += 1
        print(f"[prepare] {fname:55s} {kept:>7,}/{len(raw):>7,} kept  [{cat}]")
    return by_category

def mix_datasets(by_category, seed=1234, cap_total=500_000):
    rng = random.Random(seed)
    for cat in by_category:
        rng.shuffle(by_category[cat])

    total = sum(len(v) for v in by_category.values())

    for cat in by_category:
        print(f"[prepare] pool '{cat}': {len(by_category[cat]):,} records")

    if total <= cap_total:
        mixed = [m for pool in by_category.values() for m in pool]
        rng.shuffle(mixed)
        print(f"[prepare] total under cap ({total:,} <= {cap_total:,}), using all")
        return mixed

    take = {cat: 0 for cat in by_category}
    for cat in by_category:
        target = int(cap_total * CATEGORY_WEIGHTS[cat])
        take[cat] = min(target, len(by_category[cat]))

    slack = cap_total - sum(take.values())
    while slack > 0:
        candidates = [c for c in by_category if take[c] < len(by_category[c])]
        if not candidates:
            break
        wsum = sum(CATEGORY_WEIGHTS[c] for c in candidates)
        progressed = 0
        for c in candidates:
            bonus = int(slack * CATEGORY_WEIGHTS[c] / wsum)
            can = min(bonus, len(by_category[c]) - take[c], slack)
            if can > 0:
                take[c] += can
                progressed += can
                slack -= can
        if progressed == 0:
            break

    mixed = []
    for cat in by_category:
        mixed.extend(by_category[cat][:take[cat]])
        print(f"[prepare] '{cat}': taking {take[cat]:,}/{len(by_category[cat]):,}")
    rng.shuffle(mixed)
    return mixed

def render_chatml_ids(messages, tok, special):
    im_start, im_end = special["im_start"], special["im_end"]
    ids, loss = [], []
    system_done = False
    for m in messages:
        role, content = m["role"], m["content"].strip()
        if role == "system" and not system_done:
            chunk = [im_start] + tok.encode(f"system\n{content}", add_special_tokens=False) + [im_end]
            ids += chunk
            loss += [0] * len(chunk)
            system_done = True
            continue
        if role == "system":
            continue
        if not system_done:
            chunk = [im_start] + tok.encode(f"system\n{DEFAULT_SYSTEM_PROMPT}", add_special_tokens=False) + [im_end]
            ids += chunk
            loss += [0] * len(chunk)
            system_done = True

        role_ids = tok.encode(f"{role}\n", add_special_tokens=False)
        content_ids = tok.encode(content, add_special_tokens=False)
        chunk = [im_start] + role_ids + content_ids + [im_end]
        ids += chunk
        if role == "assistant":
            loss += [0] * (1 + len(role_ids)) + [1] * (len(content_ids) + 1)
        else:
            loss += [0] * len(chunk)
    return ids, loss

def tokenize_records(records, tok, special):
    from tqdm import tqdm
    stream, mask = [], []
    for msgs in tqdm(records, desc="tokenizing", unit="doc"):
        ids, lm = render_chatml_ids(msgs, tok, special)
        stream.extend(ids)
        mask.extend(lm)
    return stream, mask

def pack_and_save(stream, loss_mask, block_size, data_dir, val_fraction):
    n_blocks = len(stream) // block_size
    if n_blocks < 2:
        raise ValueError(f"Only {n_blocks} full blocks — not enough data.")
    arr = np.array(stream[: n_blocks * block_size], dtype=np.uint32).reshape(n_blocks, block_size)
    mask = np.array(loss_mask[: n_blocks * block_size], dtype=np.uint8).reshape(n_blocks, block_size)
    # perm = np.random.default_rng(1234).permutation(n_blocks)
    # arr, mask = arr[perm], mask[perm]
    n_val = max(1, int(n_blocks * val_fraction))
    val, train = arr[:n_val], arr[n_val:]
    val_mask, train_mask = mask[:n_val], mask[n_val:]
    os.makedirs(data_dir, exist_ok=True)
    train.tofile(os.path.join(data_dir, "train.bin"))
    val.tofile(os.path.join(data_dir, "val.bin"))
    train_mask.tofile(os.path.join(data_dir, "train_mask.bin"))
    val_mask.tofile(os.path.join(data_dir, "val_mask.bin"))
    live = int(train_mask.sum())
    print(f"[prepare] train.bin: {train.shape}, val.bin: {val.shape}")
    print(f"[prepare] loss mask: {live / train_mask.size * 100:.1f}% assistant tokens")
    print(f"[prepare] total tokens: {len(stream):,} | blocks: {n_blocks:,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=None)
    # ap.add_argument("--val-fraction", type=float, default=0.002)
    ap.add_argument("--data-dir", type=str, default=None)
    # ap.add_argument("--cap-total", type=int, default=500_000)
    
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--cap-total", type=int, default=50_000)
    args = ap.parse_args()

    from fanoos import FanoosConfig
    cfg = FanoosConfig()
    block_size = args.block_size or cfg.block_size
    data_dir = args.data_dir or os.path.join(PROJECT_ROOT, cfg.data_dir)

    tok, special = load_tokenizer()
    print(f"[prepare] tokenizer vocab: {len(tok):,}")
    print(f"[prepare] loading from: {DATA_ROOT}")

    by_category = load_all()
    records = mix_datasets(by_category, cap_total=args.cap_total)
    print(f"[prepare] mixed {len(records):,} records total")

    stream, loss_mask = tokenize_records(records, tok, special)
    pack_and_save(stream, loss_mask, block_size, data_dir, args.val_fraction)
    print("[prepare] done.")

if __name__ == "__main__":
    main()