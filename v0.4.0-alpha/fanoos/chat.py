from __future__ import annotations
import os
from typing import Iterable, Optional
from transformers import AutoTokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"

CHATML_TOKENS = [IM_START, IM_END, PAD]

DEFAULT_SYSTEM_PROMPT = "You are Fanoos AI, a helpful and knowledgeable assistant."

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)

def load_tokenizer(tokenizer_dir: str = "tokenizer", save_extended: bool = True):
    tokenizer_dir = _resolve(tokenizer_dir)
    root = _resolve(".")

    if os.path.isfile(os.path.join(tokenizer_dir, "tokenizer_config.json")):
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        return tok, _collect_ids(tok)

    raw_dir = os.path.join(_PROJECT_ROOT, "assets", "tokenizer")
    tok = AutoTokenizer.from_pretrained(raw_dir, trust_remote_code=True)

    missing = [t for t in CHATML_TOKENS if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        num_added = tok.add_special_tokens({"additional_special_tokens": missing})
        print(f"[chat] added {num_added} ChatML tokens to the tokenizer: {missing}")

    if save_extended:
        tok.save_pretrained(tokenizer_dir)
        print(f"[chat] extended tokenizer cached in {tokenizer_dir}")
    return tok, _collect_ids(tok)

def _collect_ids(tok) -> dict[str, int]:
    def tid(token: str, fallback: str) -> int:
        idx = tok.convert_tokens_to_ids(token)
        if idx == tok.unk_token_id or idx is None:
            idx = tok.convert_tokens_to_ids(fallback)
        return idx

    return {
        "im_start": tid(IM_START, IM_START),
        "im_end": tid(IM_END, IM_END),
        "pad": tid(PAD, PAD),
        "eos": tid(IM_END, "<|endoftext|>"),
    }

def format_message(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content.strip()}{IM_END}"

def build_prompt(
    messages: Iterable[dict[str, str]],
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    parts: list[str] = []
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and system_prompt:
        parts.append(format_message("system", system_prompt))
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(format_message("system", content))
        elif role == "assistant":
            parts.append(format_message("assistant", content))
        else:
            parts.append(format_message("user", content))
    text = "".join(parts)
    if add_generation_prompt:
        text += f"{IM_START}assistant\n"
    return text

def encode_chat(
    messages: Iterable[dict[str, str]],
    tok=None,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> list[int]:
    if tok is None:
        tok, _ = load_tokenizer()
    ids = _special_map(tok)
    tokens: list[int] = []
    for chunk in _iter_chatml_chunks(messages, system_prompt, add_generation_prompt):
        if chunk in ids:
            tokens.append(ids[chunk])
        else:
            tokens.extend(tok.encode(chunk, add_special_tokens=False))
    return tokens

def _iter_chatml_chunks(messages, system_prompt, add_generation_prompt):
    import re

    prompt = build_prompt(messages, system_prompt, add_generation_prompt)
    pattern = re.compile(rf"({re.escape(IM_START)}|{re.escape(IM_END)})")
    for chunk in pattern.split(prompt):
        if chunk:
            yield chunk

def _special_map(tok) -> dict[str, int]:
    ids = tok.convert_tokens_to_ids(CHATML_TOKENS)
    return dict(zip(CHATML_TOKENS, ids))

def decode_tokens(tok, token_ids: Iterable[int]) -> str:
    token_ids = list(token_ids)
    _, sp = load_tokenizer() if tok is None else (tok, _collect_ids(tok))
    if sp["im_end"] in token_ids:
        token_ids = token_ids[: token_ids.index(sp["im_end"])]
    return tok.decode(token_ids, skip_special_tokens=True).strip()

def make_chat_pair(user_text: str, assistant_text: str, system_prompt: Optional[str] = None):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})
    msgs.append({"role": "assistant", "content": assistant_text})
    return msgs

if __name__ == "__main__":
    tok, sp = load_tokenizer()
    print("special ids:", sp, "| true vocab:", len(tok))
    demo = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "Write a haiku about lanterns."},
        {"role": "assistant", "content": "Paper wings aglow / a small sun against the dark / night learns to be kind."},
    ]
    text = build_prompt(demo)
    print("--- ChatML ---")
    print(text)
    ids = encode_chat(demo, tok)
    print("--- token ids (first 30) ---", ids[:30])
    assert text == tok.decode(ids), "round-trip failed"
    print("round-trip OK, total tokens:", len(ids))