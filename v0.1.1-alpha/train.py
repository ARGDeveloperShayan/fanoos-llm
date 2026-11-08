import torch
import torch.nn as nn
from tqdm import tqdm
import os
from config import (
    BLOCK_SIZE, BATCH_SIZE, MAX_ITERS, EVAL_INTERVAL, LEARNING_RATE,
    DEVICE, EVAL_ITERS, MODEL_DIR, CHECKPOINT_PREFIX
)
from model import CharGPT

os.makedirs(MODEL_DIR, exist_ok=True)

data_path = os.path.join(MODEL_DIR, "knowledge.txt")
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {vocab_size}")
print(f"Dataset length: {len(text)} characters")
print(f"Using device: {DEVICE}")

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = CharGPT(vocab_size).to(DEVICE)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

pbar = tqdm(range(MAX_ITERS), desc="Training")
for iter in pbar:
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        pbar.set_postfix(train_loss=losses['train'].item(), val_loss=losses['val'].item())

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_model_path = os.path.join(MODEL_DIR, f"{CHECKPOINT_PREFIX}.pt")
torch.save(model.state_dict(), final_model_path)
torch.save(stoi, os.path.join(MODEL_DIR, "stoi.pt"))
torch.save(itos, os.path.join(MODEL_DIR, "itos.pt"))
print(f"Model saved to {final_model_path}")

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
sample_output = decode(model.generate(context, max_new_tokens=200)[0].tolist())
sample_output = sample_output.replace('|', ' ').strip()
print("\n--- Sample generation after training ---")
print(sample_output)