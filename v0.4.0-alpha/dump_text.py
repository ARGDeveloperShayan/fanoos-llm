import itertools, os
import pyarrow as pa, pyarrow.parquet as pq
from datasets import load_dataset
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "datasets")
JOBS = [("cosmopedia-v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 600_000),
        ("python-code", "codeparrot/codeparrot-clean", None, "content", 150_000)]
for prefix, repo, cfg, field, n in JOBS:
    ds = load_dataset(repo, cfg, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    buf, k = [], 0
    for row in itertools.islice(ds, n):
        t = row[field]
        if t and len(t) > 200: buf.append(t)
        if len(buf) >= 100_000:
            pq.write_table(pa.table({"text": buf}), os.path.join(OUT, f"{prefix}-{k:05d}.parquet")); buf, k = [], k + 1
    if buf: pq.write_table(pa.table({"text": buf}), os.path.join(OUT, f"{prefix}-{k:05d}.parquet"))