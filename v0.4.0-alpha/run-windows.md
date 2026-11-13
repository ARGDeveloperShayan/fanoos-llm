# Fanoos AI: run guide (PowerShell, Windows)

## Step 0. Install
 
```powershell
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Step 1. Data layout
 
`assets/datasets/` must contain:
- `fineweb-edu-1B-*.parquet`
- `wikipedia-20231101-en-*.parquet`
- `smoltalk-all-*.parquet`
- `python-code-*.parquet`
- `sft_conversation.jsonl`
- `sft_knowledge.jsonl`
- `sft_reasoning.jsonl`
- `sft_code.jsonl`
- `identity.jsonl`

Regenerate **only** the `identity.jsonl` file (do not run the whole `download_data.py`, it re-downloads everything):
 
```powershell
Remove-Item assets\datasets\identity.jsonl -ErrorAction SilentlyContinue
python -c "import download_data as d; d.dump_identity()"
```

## Step 2. Tokenizer (first time only)
 
```powershell
python tokenizer.py --vocab-size 32000 --cap-records 100000
python -c "from fanoos import load_tokenizer; t,_=load_tokenizer(); print('vocab:', len(t))"
```

Expected output: `vocab: 32000`.

## Step 3. Prepare pretrain data
 
```powershell
python prepare_pretrain.py
```
 
Output: `data_pretrain/` (about 1.3B tokens). Per-source caches go to `assets/pretrain_bins/` and are reused on re-runs.

## Step 4. Pretrain (about 5.5 h at ~1.6 s/step)
 
```powershell
python train.py --preset fanoos-130m --data-dir data_pretrain --ckpt-dir checkpoints_pretrain `
  --micro-batch 16 --grad-accum 8 --max-iters 12000 `
  --lr 6e-4 --min-lr 6e-5 --warmup 300 --dropout 0 --compile --no-ckpt
```
 
- Do **not** pass `--aligned` here.
- Healthy run: step-0 loss about 10.5, final val loss about 3.3 (ppl about 27).
- Resume after an interruption: add `--resume`.
- Outputs: `best.pt` (lowest val), `ckpt.pt` (resumable), `final.pt`.

## Step 5. Sanity-check the base model
 
Run `base_sample.py` (it continues text, it does not chat). The output should be fluent English even if the facts are shaky.
 
```powershell
python base_sample.py
```
 
If this is gibberish, stop. Fix pretraining before doing SFT.
 
## Step 6. Prepare SFT data
 
```powershell
python prepare.py --cap-total 150000 --block-size 512 --data-dir data_sft
```
 
Check the log:
- `identity.jsonl` should keep roughly all of its lines (about 1,500).
- Records per category should roughly follow 30% conversation, 45% knowledge, 8% reasoning, 17% code.
- Dropped-over-length records should be small.

## Step 7. SFT (about 45 min)
 
```powershell
python train.py --preset fanoos-130m --data-dir data_sft --ckpt-dir checkpoints_sft `
  --init-from checkpoints_pretrain/best.pt --aligned `
  --micro-batch 16 --grad-accum 4 --max-iters 3000 `
  --lr 1e-4 --min-lr 1e-5 --warmup 100 --dropout 0.05 --eval-iters 50 --compile --no-ckpt
```
 
Verify immediately in the first lines of the log:
- `[train] init from checkpoints_pretrain/best.pt OK` appears.
- Step-0 loss is **well below 10** (roughly 3-5). If it is about 10.5, the pretrained weights did NOT load.
- `--aligned` is required for SFT so every training window starts at a document boundary.

## Step 8. Evaluate and serve
 
```powershell
python benchmark.py --checkpoint checkpoints_sft/best.pt --json-out benchmark.json
python test.py --checkpoint checkpoints_sft/best.pt --temperature 0.6 --top-p 0.9
python serve.py --checkpoint checkpoints_sft/best.pt --port 8000
```
 
- Open http://127.0.0.1:8000. Use `--host 0.0.0.0` to expose it on your network.
- Note: `serve.py` samples with `model._sample` directly, so the `repetition_penalty` in `generate()` does not apply there. Use `temperature` around 0.6 and `top_p` around 0.9 in your requests.
---
 
## Step 9. Optional: stronger base model (about 9 h)
 
Add more pretraining text without touching the tokenizer.
 
```powershell
python dump_text.py
```

__*writes cosmopedia-v2 and python-code*__
 
Edit `SOURCES` in `prepare_pretrain.py` to:
 
```python
SOURCES = [
    ("fineweb", "fineweb-edu-1B-*.parquet",        "text", 0.45),
    ("wiki",    "wikipedia-20231101-en-*.parquet", "text", 0.15),
    ("cosmo",   "cosmopedia-v2-*.parquet",         "text", 0.25),
    ("python",  "python-code-*.parquet",           "text", 0.10),
]
```
 
Then:
 
```powershell
python prepare_pretrain.py
 
python train.py --preset fanoos-130m --data-dir data_pretrain --ckpt-dir checkpoints_pretrain2 `
  --init-from checkpoints_pretrain/best.pt `
  --micro-batch 16 --grad-accum 8 --max-iters 20000 `
  --lr 3e-4 --min-lr 3e-5 --warmup 100 --dropout 0 --compile --no-ckpt
 
# redo SFT from the new base, into a NEW folder so the old SFT model is kept
python train.py --preset fanoos-130m --data-dir data_sft --ckpt-dir checkpoints_sft2 `
  --init-from checkpoints_pretrain2/best.pt --aligned `
  --micro-batch 16 --grad-accum 4 --max-iters 3000 `
  --lr 1e-4 --min-lr 1e-5 --warmup 100 --dropout 0.05 --eval-iters 50 --compile --no-ckpt
 
python test.py --checkpoint checkpoints_sft2/best.pt --temperature 0.6 --top-p 0.9
```
 
---
 
## Optional: export
 
```powershell
python export_hf.py --checkpoint checkpoints_sft/best.pt --out-dir export\fanoos-ai
python export_gguf.py --hf-dir export\fanoos-ai --out-dir export\gguf --llama-cpp-dir C:\path\to\llama.cpp --outtype q8_0
```

## Troubleshooting
 
| Symptom | Cause / fix |
|---|---|
| SFT step-0 loss about 10.5 | Weights did not load. The `train.py` patch (`clean_state_dict`) is missing. Checkpoints saved with `--compile` have `_orig_mod.` key prefixes. |
| `Some tensors share memory` at end of training | `export_hf` got the compiled model. Use `raw_model` in `train.py`. `final.pt` is already saved, so nothing is lost. |
| CUDA out of memory | `--micro-batch 8 --grad-accum 16` (same effective batch). |
| `prepare_pretrain.py` says "no text column" | That parquet has no `text`/`content` column. Skip it or add its column name to `detect_text_column`. |
| Model answers greetings but rambles on other prompts | Check SFT mix in Step 6 and that `--aligned` was used. |
| Torch graph-break warnings from `.item()` | Fixed by `rope.get_range(...)` in `model.py`. |