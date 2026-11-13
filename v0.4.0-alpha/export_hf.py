from __future__ import annotations
import argparse
import os
import shutil
import torch
from safetensors.torch import save_file
from fanoos import FanoosConfig, FanoosLLM, load_tokenizer, clean_state_dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser(description="Export Fanoos checkpoint to HF format")
    ap.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = FanoosConfig(**ckpt["cfg"])

    tok, _ = load_tokenizer()
    cfg.resolve_vocab_size(len(tok))

    model = FanoosLLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    save_file(state, os.path.join(args.out_dir, "model.safetensors"))
    print(f"[export] model.safetensors saved")

    cfg.save(os.path.join(args.out_dir, "config.json"))
    print(f"[export] config.json saved")

    tok_src = os.path.join(PROJECT_ROOT, cfg.tokenizer_dir)
    if os.path.isdir(tok_src):
        for fname in os.listdir(tok_src):
            if fname != "chatml_special_tokens.json":
                shutil.copy2(os.path.join(tok_src, fname), os.path.join(args.out_dir, fname))
        print(f"[export] tokenizer files copied")

    print(f"[export] Verifying load...")
    import json
    with open(os.path.join(args.out_dir, "config.json")) as f:
        loaded_cfg = json.load(f)
    print(f"[export] Config: {loaded_cfg['n_layer']}L, {loaded_cfg['n_head']}H, {loaded_cfg['n_embd']}D, vocab={loaded_cfg['vocab_size']}")

    print(f"[export] Done! Output: {args.out_dir}")

if __name__ == "__main__":
    main()