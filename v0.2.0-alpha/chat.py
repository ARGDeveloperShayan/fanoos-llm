import torch
import os
import sys
import pickle
from model import GPTConfig, GPT

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

meta_path = os.path.join('data', 'tinyllm', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    return [stoi.get(c, 0) for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

# ---------- Chat UI ----------
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_help():
    print("\n" + "="*50)
    print("nanoGPT Chat – TinyLLM")
    print("Commands: /exit, /reload, /clear, /temp <value>")
    print("="*50 + "\n")

# ---------- Load initial model ----------
ckpt = os.path.join('tinyllm', 'ckpt.pt')
model, config = load_model(ckpt)
temperature = 0.8

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
        print("Model loaded.")
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
            if next_id.item() == stoi.get('\n', -1):
                break

    full_output = decode(x[0].tolist())
    response = full_output[len(prompt):].strip()
    if '\n' in response:
        response = response.split('\n')[0]
    response = response.replace("Assistant:", "").strip()

    print(f"AI: {response}")