import os
import sys
import re
import pickle
import torch
import tiktoken

from model   import GPTConfig, GPT
from tools   import process_tools


# ──────────────────────────────────────────────────────────────────────────────
#  GPT-2 BPE offline tokeniser
# ──────────────────────────────────────────────────────────────────────────────
def load_gpt2_encoding_offline(encoder_json_path, vocab_bpe_path):
    with open(vocab_bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    merges = [tuple(l.split()) for l in bpe_data.strip().split("\n")[1:]]
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    for rank, (t1, t2) in enumerate(merges):
        mergeable_ranks[t1.encode() + t2.encode()] = 256 + rank
    return tiktoken.Encoding(
        name="gpt2",
        pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": 50256},
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Device
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")


# ──────────────────────────────────────────────────────────────────────────────
#  Tokeniser setup
# ──────────────────────────────────────────────────────────────────────────────
meta_path = os.path.join("data", "meta.pkl")

if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if "stoi" in meta and "itos" in meta:
        stoi, itos = meta["stoi"], meta["itos"]
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda ids: "".join(itos[i] for i in ids)
        newline_token = stoi.get("\n", -1)
        tokeniser_type = "char"
    else:
        enc = load_gpt2_encoding_offline(
            "data/gpt2_encoder/encoder.json",
            "data/gpt2_encoder/vocab.bpe",
        )
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda ids: enc.decode(ids)
        newline_token = enc.encode_single_token("\n")
        tokeniser_type = "bpe"
else:
    enc = load_gpt2_encoding_offline(
        "data/gpt2_encoder/encoder.json",
        "data/gpt2_encoder/vocab.bpe",
    )
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)
    newline_token = enc.encode_single_token("\n")
    tokeniser_type = "bpe"

print(f"Tokeniser: {tokeniser_type}")


# ──────────────────────────────────────────────────────────────────────────────
#  Model loader
# ──────────────────────────────────────────────────────────────────────────────
CKPT_PATH = os.path.join("model", "ckpt.pt")

def load_model(path: str):
    if not os.path.exists(path):
        print(f"[Error] Checkpoint not found: {path}")
        sys.exit(1)
    ckpt   = torch.load(path, map_location=device)
    config = GPTConfig(**ckpt["model_args"])
    model  = GPT(config)
    state  = ckpt["model"]
    state  = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


# ──────────────────────────────────────────────────────────────────────────────
#  Conversation history  (memory across turns)
# ──────────────────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6   # how many past user/assistant pairs to include in prompt

history: list[tuple[str, str]] = []


def build_prompt(user_input: str) -> str:
    parts = []
    for user_turn, assistant_turn in history[-MAX_HISTORY_TURNS:]:
        parts.append(f"User: {user_turn}\nAssistant: {assistant_turn}")
    parts.append(f"User: {user_input}\nAssistant:")
    return "\n".join(parts)


def add_to_history(user_input: str, response: str):
    history.append((user_input, response))
    if len(history) > MAX_HISTORY_TURNS * 2:
        history.pop(0)


# ──────────────────────────────────────────────────────────────────────────────
#  Generation helper
# ──────────────────────────────────────────────────────────────────────────────
def generate_response(
    model: GPT,
    config: GPTConfig,
    prompt: str,
    max_tokens: int = 120,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    use_tools: bool = True,
) -> str:
    input_ids     = encode(prompt)
    x             = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_p = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_p - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            x = torch.cat((x, torch.tensor([[next_id]], device=device)), dim=1)
            generated_ids.append(next_id)

            if next_id == newline_token:
                break

    full_text = decode(generated_ids)
    response  = full_text[len(prompt):]
    response  = response.lstrip()
    response  = re.sub(r"^Assistant\s*:\s*", "", response)
    response  = response.strip()

    if use_tools:
        response = process_tools(response)

    return response


# ──────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ──────────────────────────────────────────────────────────────────────────────
DIVIDER = "─" * 52

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def show_help():
    print(f"\n{DIVIDER}")
    print("  Fanoos AI  –  Chat Interface")
    print(DIVIDER)
    print("  Commands:")
    print("    /exit           quit")
    print("    /reload         reload checkpoint")
    print("    /clear          clear screen")
    print("    /temp <f>       sampling temperature (default 0.7)")
    print("    /top_k <n>      top-k  (0 = off,  default 50)")
    print("    /top_p <f>      nucleus threshold (default 0.95)")
    print("    /tools          toggle tool processing on / off")
    print("    /history        show conversation history")
    print("    /reset          clear conversation history")
    print("    /help           show this help")
    print(DIVIDER)
    print("  Active tools (when /tools is ON):")
    print("    [CALC:expr]     safe arithmetic evaluation")
    print("    [CLCK:UTC]      current date & time in UTC")
    print("    [CLCK:DATE/TIME/YEAR/DAY]")
    print(DIVIDER + "\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Main chat loop
# ──────────────────────────────────────────────────────────────────────────────
model, config = load_model(CKPT_PATH)

temperature = 0.7
top_k       = 50
top_p       = 0.95
tools_on    = True

clear_screen()
show_help()

while True:
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    cmd = user_input.lower()

    # ── Commands ──────────────────────────────────────────────────────────────
    if cmd.startswith("/exit"):
        print("Goodbye!")
        break

    elif cmd.startswith("/clear"):
        clear_screen()
        show_help()
        continue

    elif cmd.startswith("/reload"):
        print("Reloading model …")
        model, config = load_model(CKPT_PATH)
        print("Model reloaded.")
        continue

    elif cmd.startswith("/help"):
        show_help()
        continue

    elif cmd.startswith("/tools"):
        tools_on = not tools_on
        state = "ON  ✓ (tool tags will be evaluated)" if tools_on else "OFF ✗ (raw model output shown)"
        print(f"Tools: {state}")
        continue

    elif cmd.startswith("/history"):
        if not history:
            print("  (no history yet)")
        else:
            print(f"\n{DIVIDER}")
            for i, (u, a) in enumerate(history, 1):
                print(f"  [{i}] You:    {u}")
                print(f"       Fanoos: {a}")
            print(DIVIDER)
        continue

    elif cmd.startswith("/reset"):
        history.clear()
        print("Conversation history cleared.")
        continue

    elif cmd.startswith("/temp"):
        parts = user_input.split()
        if len(parts) == 2:
            try:
                temperature = float(parts[1])
                print(f"Temperature → {temperature}")
            except ValueError:
                print("Usage: /temp <float>  e.g.  /temp 0.8")
        else:
            print(f"Current temperature: {temperature}")
        continue

    elif cmd.startswith("/top_k"):
        parts = user_input.split()
        if len(parts) == 2:
            try:
                top_k = int(parts[1])
                print(f"Top-k → {top_k}  (0 = disabled)")
            except ValueError:
                print("Usage: /top_k <int>  e.g.  /top_k 40")
        else:
            print(f"Current top_k: {top_k}")
        continue

    elif cmd.startswith("/top_p"):
        parts = user_input.split()
        if len(parts) == 2:
            try:
                top_p = float(parts[1])
                print(f"Top-p → {top_p}")
            except ValueError:
                print("Usage: /top_p <float>  e.g.  /top_p 0.9")
        else:
            print(f"Current top_p: {top_p}")
        continue

    # ── Normal message ─────────────────────────────────────────────────────────
    prompt   = build_prompt(user_input)
    response = generate_response(
        model, config, prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_tools=tools_on,
    )

    add_to_history(user_input, response)

    tools_note = "" if tools_on else "  [tools OFF]"
    print(f"Fanoos: {response}{tools_note}\n")