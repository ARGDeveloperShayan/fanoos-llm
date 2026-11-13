from importlib.resources import path
import itertools, json, os
import random
from datasets import load_dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
PRE  = os.path.join(ROOT, "assets", "pretrain")
SFT  = os.path.join(ROOT, "assets", "datasets")
os.makedirs(PRE, exist_ok=True)
os.makedirs(SFT, exist_ok=True)

PRETRAIN = [
    ("fineweb_edu", "codelion/fineweb-edu-1B",     None,            "text",    450_000),
    ("cosmopedia",  "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text",    250_000),
    ("wikipedia",   "wikimedia/wikipedia",         "20231101.en",   "text",    150_000),
    ("python_code", "codeparrot/codeparrot-clean", None,            "content",  80_000),
    # ("wiki_fa",     "wikimedia/wikipedia",         "20231101.fa",   "text",    100_000),
]

STATS = {"pretrain": {}, "sft": {}, "identity": {}}

def _tok(chars):      return chars / 4.0
def _gb(n_bytes):     return n_bytes / 1e9

def _is_valid_jsonl(path, min_bytes=256):
    """Cheap 'is this download complete?' check:
       - file exists, size > min_bytes
       - last non-empty line parses as JSON
    Because we write to `<path>.tmp` and `os.replace` only after the loop
    finishes, a file at the final path should already be complete. This
    just guards against external truncation / old-format leftovers."""
    if not os.path.isfile(path):
        return False
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            chunk = min(65536, end)
            f.seek(end - chunk)
            tail = f.read().decode("utf-8", errors="replace")
        last = [ln for ln in tail.splitlines() if ln.strip()][-1]
        json.loads(last)
        return True
    except Exception:
        return False

def _scan_jsonl(path):
    """Count docs + estimate content chars (only used for small skipped files)."""
    docs = chars = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs += 1
            if "text" in obj:
                chars += len(obj["text"])
            elif "messages" in obj:
                chars += sum(len(m.get("content", "")) for m in obj["messages"])
    return docs, chars

# --------------------------------------------------------------------------- #
#  Pretrain
# --------------------------------------------------------------------------- #
def dump(name, hf_id, cfg, field, n):
    out = os.path.join(PRE, f"{name}.jsonl")

    if _is_valid_jsonl(out):
        size = os.path.getsize(out)
        print(f"[download] {name}: already complete on disk ({_gb(size):.2f} GB) — skip")
        STATS["pretrain"][name] = {"docs": None, "chars": None, "bytes": size, "skipped": True}
        return

    if os.path.isfile(out):
        print(f"[download] {name}: existing file incomplete, re-downloading")
        os.remove(out)

    ds = load_dataset(hf_id, cfg, split="train", streaming=True) \
            .shuffle(seed=0, buffer_size=10_000)
    tmp = out + ".tmp"
    docs = chars = 0

    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(itertools.islice(ds, n)):
            t = row[field]
            if t and len(t) > 200:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                docs += 1
                chars += len(t)

            if i and i % 20000 == 0:
                f.flush()
                size_now = os.path.getsize(tmp)
                est_size = size_now * n / i       # linear projection
                print(f"[download] {name}: scanned {i:,}/{n:,}, wrote {docs:,} docs  "
                      f"(~{_tok(chars)/1e6:.1f}M tok, {_gb(size_now):.2f} GB so far, "
                      f"~{_gb(est_size):.2f} GB projected)", flush=True)

    os.replace(tmp, out)
    size = os.path.getsize(out)
    STATS["pretrain"][name] = {"docs": docs, "chars": chars, "bytes": size, "skipped": False}
    print(f"[download] {name}: DONE  {docs:,} docs, "
          f"~{_tok(chars)/1e6:.1f}M tokens, {_gb(size):.2f} GB", flush=True)

# --------------------------------------------------------------------------- #
#  SFT
# --------------------------------------------------------------------------- #
CATS = {
    "conversation": {"everyday-conversations": 3000,  "systemchats-30k":    30000, "smol-constraints": 30000},
    "knowledge":    {"smol-magpie-ultra":      90000, "openhermes-100k":    40000,
                     "smol-rewrite":           25000, "smol-summarize":     25000},
    "reasoning":    {"metamathqa-50k":         50000, "numina-cot-100k":    40000},
    "code":         {"self-oss-instruct":      30000},
}

def dump_sft():
    src2cat     = {s: c for c, d in CATS.items() for s in d}
    quota       = {s: q for d in CATS.values() for s, q in d.items()}
    total_quota = sum(quota.values())
    out_paths   = {c: os.path.join(SFT, f"sft_{c}.jsonl") for c in CATS}

    # All category files already present? -> skip the whole pass.
    if all(_is_valid_jsonl(p) for p in out_paths.values()):
        print("[download] sft: all category files present & valid — skip")
        total_docs = total_chars = total_bytes = 0
        for c, p in out_paths.items():
            docs, chars = _scan_jsonl(p)
            size = os.path.getsize(p)
            STATS["sft"][c] = {"docs": docs, "chars": chars, "bytes": size}
            total_docs  += docs;  total_chars += chars;  total_bytes += size
        STATS["sft"]["_total"] = {"docs": total_docs, "chars": total_chars,
                                  "bytes": total_bytes, "quota": total_quota}
        return

    # Clean stale / partial artifacts from earlier runs.
    tmp_paths = {}
    for c, p in out_paths.items():
        if os.path.isfile(p) and not _is_valid_jsonl(p):
            os.remove(p)
        tmp = p + ".tmp"
        if os.path.isfile(tmp):
            os.remove(tmp)
        tmp_paths[c] = tmp

    counts     = dict.fromkeys(quota, 0)
    cat_docs   = dict.fromkeys(CATS, 0)
    cat_chars  = dict.fromkeys(CATS, 0)
    total_docs = total_chars = 0

    files = {c: open(tmp_paths[c], "w", encoding="utf-8") for c in CATS}
    try:
        for row in load_dataset("HuggingFaceTB/smoltalk", "all",
                                split="train", streaming=True):
            s = row["source"]
            if s not in quota or counts[s] >= quota[s]:
                continue

            msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
            c = src2cat[s]
            files[c].write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

            counts[s]  += 1
            cat_docs[c] += 1
            total_docs += 1
            nchars = sum(len(m["content"]) for m in msgs)
            cat_chars[c] += nchars
            total_chars  += nchars

            if total_docs % 10000 == 0:
                for cf in files.values():
                    cf.flush()
                cur = sum(os.path.getsize(tp) for tp in tmp_paths.values()
                          if os.path.isfile(tp))
                print(f"[download] sft: {total_docs:,}/{total_quota:,} examples "
                      f"({100*total_docs/total_quota:.0f}%), {_gb(cur):.2f} GB so far",
                      flush=True)

            if all(counts[x] >= quota[x] for x in quota):
                break
    finally:
        for f in files.values():
            f.close()

    print("[download] smoltalk counts:", counts)
    for c, p in out_paths.items():
        os.replace(tmp_paths[c], p)
        STATS["sft"][c] = {
            "docs":  cat_docs[c],
            "chars": cat_chars[c],
            "bytes": os.path.getsize(p),
        }
    total_bytes = sum(STATS["sft"][c]["bytes"] for c in CATS)
    STATS["sft"]["_total"] = {"docs": total_docs, "chars": total_chars,
                              "bytes": total_bytes, "quota": total_quota}
    print(f"[download] sft: DONE  {total_docs:,}/{total_quota:,} examples, "
          f"~{_tok(total_chars)/1e6:.1f}M tokens, {_gb(total_bytes):.2f} GB", flush=True)

# --------------------------------------------------------------------------- #
#  Identity
# --------------------------------------------------------------------------- #
NAME, CREATOR = "Fanoos AI", "Shayan"
TOPICS = [
 (["What's your name?", "What is your name?", "Who are you?", "Tell me your name.", "Introduce yourself.", "What are you called?"],
  [f"I'm {NAME}, a small AI language model created by {CREATOR}.", f"My name is {NAME}. I'm a small language model made by {CREATOR}."]),
 (["Who created you?", "Who made you?", "Who built you?", "Who is your creator?"],
  [f"I was created by {CREATOR}.", f"{CREATOR} created me. I'm {NAME}."]),
 (["Are you an AI?", "Are you an AI model?", "Are you a human?", "Are you a robot?"],
  [f"I'm an AI language model called {NAME}, not a human. I can chat and answer questions."]),
 (["What can you do?", "How can you help me?", "What are you good at?"],
  ["I can chat, answer general questions, explain concepts and help with simple Python code. I'm a small model, so I can make mistakes."]),
 (["Hi, how are you?", "Hello, how are you doing?", "How are you today?", "Hey, how's it going?"],
  ["I'm doing well, thanks for asking! How can I help you today?", "I'm good, thank you! What can I do for you?"]),
 (["Hello!", "Hi!", "Hey!", "Hi there!", "Good morning!"],
  ["Hello! How can I help you today?", "Hi there! What can I do for you?"]),
 (["Nice to meet you!", "Pleased to meet you."], ["Nice to meet you too! How can I help you today?"]),
 (["Thank you for your help.", "Thanks!", "Thank you so much!"],
  ["You're welcome! Let me know if you need anything else.", "Happy to help! Anything else?"]),
 (["Goodbye!", "Bye!", "See you later!"], ["Goodbye! Have a great day.", "Bye! Come back any time."]),
]

def dump_identity():
    path = os.path.join(SFT, "identity.jsonl")
    if _is_valid_jsonl(path):
        docs, chars = _scan_jsonl(path)
        size = os.path.getsize(path)
        print(f"[download] identity: already complete on disk "
              f"({_gb(size)*1e3:.1f} MB) — skip")
        STATS["identity"] = {"docs": docs, "chars": chars, "bytes": size}
        return

    tmp  = path + ".tmp"
    docs = chars = 0
    rng = random.Random(0)
    with open(tmp, "w", encoding="utf-8") as f:
        for _ in range(40):
            for qs, ans in TOPICS:
                for q in qs:
                    a = rng.choice(ans)
                    f.write(json.dumps({"messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]}) + "\n")
                    docs += 1
                    chars += len(q) + len(a)
    os.replace(tmp, path)

    size = os.path.getsize(path)
    STATS["identity"] = {"docs": docs, "chars": chars, "bytes": size}
    print(f"[download] identity: DONE  {docs:,} lines, "
          f"~{_tok(chars)/1e3:.1f}K tokens, {size/1e3:.1f} KB", flush=True)

# --------------------------------------------------------------------------- #
#  Summary
# --------------------------------------------------------------------------- #
def print_summary():
    print("\n" + "=" * 74)
    print("DOWNLOAD SUMMARY")
    print("=" * 74)

    print("Pretrain (assets/pretrain):")
    p_docs = p_chars = p_bytes = 0
    p_skipped = 0
    for name, info in STATS["pretrain"].items():
        if info["docs"] is None:
            tok, docs = "existing", "-"
            p_skipped += 1
        else:
            tok  = f"~{_tok(info['chars'])/1e6:.1f}M tok"
            docs = f"{info['docs']:,} docs"
            p_docs  += info["docs"]
            p_chars += info["chars"]
        p_bytes += info["bytes"]
        print(f"  {name:<14} {docs:>16}  {tok:>14}  {_gb(info['bytes']):>8.2f} GB")
    note = "  (some files were skipped; counts unknown)" if p_skipped else ""
    print(f"  {'SUBTOTAL':<14} {p_docs:>16,}  ~{_tok(p_chars)/1e6:>8.1f}M tok  "
          f"{_gb(p_bytes):>8.2f} GB{note}")

    print("\nSFT (assets/datasets):")
    for c, info in STATS["sft"].items():
        if c == "_total":
            continue
        print(f"  {c:<14} {info['docs']:>16,}  ~{_tok(info['chars'])/1e6:>8.1f}M tok  "
              f"{_gb(info['bytes']):>8.2f} GB")
    t = STATS["sft"].get("_total", {})
    if t:
        print(f"  {'SUBTOTAL':<14} {t['docs']:>16,}  ~{_tok(t['chars'])/1e6:>8.1f}M tok  "
              f"{_gb(t['bytes']):>8.2f} GB  (quota {t['quota']:,})")

    i = STATS["identity"]
    print("\nIdentity:")
    print(f"  {'identity':<14} {i.get('docs', 0):>16,}  "
          f"~{_tok(i.get('chars', 0))/1e3:>8.1f}K tok  {i.get('bytes', 0)/1e3:>8.1f} KB")

    g_docs  = p_docs  + t.get("docs",  0) + i.get("docs",  0)
    g_chars = p_chars + t.get("chars", 0) + i.get("chars", 0)
    g_bytes = p_bytes + t.get("bytes", 0) + i.get("bytes", 0)
    print("\n" + "-" * 74)
    print(f"GRAND TOTAL: {g_docs:,} items, ~{_tok(g_chars)/1e6:.1f}M tokens, "
          f"{_gb(g_bytes):.2f} GB on disk")
    print("-" * 74)

if __name__ == "__main__":
    dump_identity()
    dump_sft()
    for p in PRETRAIN:
        dump(*p)
    print_summary()