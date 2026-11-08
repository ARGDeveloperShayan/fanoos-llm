import torch
import os
import sys
import pickle
import tiktoken
import json
from model import GPTConfig, GPT

def load_gpt2_encoding_offline(encoder_json_path, vocab_bpe_path):
    with open(vocab_bpe_path, 'r', encoding='utf-8') as f:
        bpe_data = f.read()
    merges = [tuple(line.split()) for line in bpe_data.strip().split('\n')[1:]]

    mergeable_ranks = {bytes([i]): i for i in range(256)}

    for rank, (t1, t2) in enumerate(merges):
        key = t1.encode('utf-8') + t2.encode('utf-8')
        mergeable_ranks[key] = 256 + rank

    special_tokens = {"<|endoftext|>": 50256}

    return tiktoken.Encoding(
        name="gpt2",
        pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- Load model ----------
def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found!")
        sys.exit(1)
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, config

# ---------- Tokenizer setup ----------
meta_path = os.path.join('data', 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if 'stoi' in meta and 'itos' in meta:
        stoi, itos = meta['stoi'], meta['itos']
        def encode(s):
            return [stoi.get(c, 0) for c in s]
        def decode(ids):
            return ''.join([itos[i] for i in ids])
        newline_token = stoi.get('\n', -1)
        tokenizer_type = 'char'
    else:
        enc = load_gpt2_encoding_offline(
            "data/gpt2_encoder/encoder.json",
            "data/gpt2_encoder/vocab.bpe"
        )
        def encode(s):
            return enc.encode(s, allowed_special={"<|endoftext|>"})
        def decode(ids):
            return enc.decode(ids)
        newline_token = enc.encode_single_token('\n')
        tokenizer_type = 'bpe'
else:
    enc = load_gpt2_encoding_offline(
        "data/gpt2_encoder/encoder.json",
        "data/gpt2_encoder/vocab.bpe"
    )
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ids: enc.decode(ids)
    newline_token = enc.encode_single_token('\n')
    tokenizer_type = 'bpe'

print(f"Tokenizer type: {tokenizer_type}")

# ---------- Chat UI ----------
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_help():
    print("\n" + "="*50)
    print("TinyLLM Chat (BPE)")
    print("Commands: /exit, /reload, /clear, /temp <value>")
    print("="*50 + "\n")

# ---------- Load initial model ----------
ckpt = os.path.join('model', 'ckpt.pt')
model, config = load_model(ckpt)
temperature = 0.7

clear_screen()
show_help()

while True:
    user_input = input("\nYou: ").strip()
    if user_input.startswith("/exit"):
        break
    elif user_input.startswith("/clear"):
        clear_screen()
        show_help()
        continue
    elif user_input.startswith("/reload"):
        model, _ = load_model(ckpt)
        print("Model reloaded.")
        continue
    elif user_input.startswith("/temp"):
        try:
            temperature = float(user_input.split()[1])
            print(f"Temperature set to {temperature}")
        except:
            print("Usage: /temp <value>")
        continue
    elif not user_input:
        continue

    prompt = f"User: {user_input}\nAssistant:"
    input_ids = encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(100):
            x_cond = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            x = torch.cat((x, next_id), dim=1)
            if next_id.item() == newline_token:
                break

    full_output = decode(x[0].tolist())
    response = full_output[len(prompt):].strip()
    if '\n' in response:
        response = response.split('\n')[0]
    response = response.replace("Assistant:", "").strip()

    print(f"AI: {response}")