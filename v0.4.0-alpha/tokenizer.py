from __future__ import annotations
import argparse
import os
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "assets", "datasets")

from prepare import DATASETS, load_file, adapt_record

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|pad|>", "<|endoftext|>"]

def iter_text(cap_records: int = 50000):
    import glob, json
    for p in sorted(glob.glob(os.path.join(PROJECT_ROOT, "assets", "pretrain", "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20000: break
                yield json.loads(line)["text"][:4000]
    per_file = max(1, cap_records // len(DATASETS))
    for fname, _cat in DATASETS:
        path = os.path.join(DATA_ROOT, fname)
        if not os.path.isfile(path): continue
        n = 0
        for rec in load_file(path):
            msgs = adapt_record(rec)
            if not msgs: continue
            yield "\n".join(m["content"] for m in msgs)
            n += 1
            if n >= per_file: break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--cap-records", type=int, default=50000)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "tokenizer"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"[tokenizer] training BPE vocab={args.vocab_size} on ≤{args.cap_records} docs…")
    tok.train_from_iterator(iter_text(args.cap_records), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=None,
        pad_token="<|pad|>",
        eos_token="<|im_end|>",
        bos_token=None,
        additional_special_tokens=SPECIAL_TOKENS,
    )
    fast.save_pretrained(args.out_dir)
    print(f"[tokenizer] saved to {args.out_dir}  |  vocab_size={fast.vocab_size}")
    ids = fast.encode("<|im_start|>user\nhi<|im_end|>", add_special_tokens=False)
    print(f"[tokenizer] round-trip: {fast.decode(ids)!r}")

if __name__ == "__main__":
    main()