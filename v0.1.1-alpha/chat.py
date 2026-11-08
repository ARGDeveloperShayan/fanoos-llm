import torch
import torch.nn.functional as F
import os
import re
from config import MODEL_DIR, CHECKPOINT_PREFIX, DEVICE, MAX_LEN, TEMPERATURE, BLOCK_SIZE
from model import CharGPT

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

stoi = torch.load(os.path.join(MODEL_DIR, "stoi.pt"), weights_only=False)
itos = torch.load(os.path.join(MODEL_DIR, "itos.pt"), weights_only=False)
vocab_size = len(stoi)

def encode(s):
    return [stoi.get(ch, stoi[' ']) for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def find_latest_checkpoint():
    final_path = os.path.join(MODEL_DIR, f"{CHECKPOINT_PREFIX}.pt")
    if os.path.exists(final_path):
        return final_path
    pattern = re.compile(rf"{re.escape(CHECKPOINT_PREFIX)}_epoch(\d+)\.pt")
    epoch_files = []
    for f in os.listdir(MODEL_DIR):
        m = pattern.match(f)
        if m:
            epoch_files.append((int(m.group(1)), f))
    if epoch_files:
        latest = max(epoch_files, key=lambda x: x[0])[1]
        return os.path.join(MODEL_DIR, latest)
    raise FileNotFoundError(f"No checkpoint found in {MODEL_DIR}")

def calculate(expression):
    """Safely evaluate math expression using Python's arithmetic."""
    expression = expression.replace(" ", "")
    if any(x in expression for x in ['+','-','*','/']):
        try:
            # Evaluate but be safe
            result = eval(expression)
            # Convert to integer if it's a whole number
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return str(result)
        except:
            return None
    return None

model = CharGPT(vocab_size).to(device)
checkpoint_path = find_latest_checkpoint()
print(f"Loading: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

def generate(prompt, max_new_tokens=30, temperature=TEMPERATURE):
    tokens = encode(prompt)
    if len(tokens) > BLOCK_SIZE:
        tokens = tokens[-BLOCK_SIZE:]
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = context[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        last_logits = logits[0, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)

        if itos[next_token] in ['\n', '|']:
            break
        if context.shape[1] >= MAX_LEN:
            break

    result = decode(context[0].tolist())
    answer = result[len(prompt):]
    for stop in ['|', '\n']:
        if stop in answer:
            answer = answer.split(stop)[0]
            break
    return answer.strip()

print("\nYour custom AI is ready! Type 'exit' to quit.\n")
while True:
    user = input("You: ").strip().lower()
    if user == "exit":
        break
    
    math_match = re.match(r'^(\d+)\s*([+\-*/])\s*(\d+)$', user)
    if math_match:
        result = calculate(user)
        if result is not None:
            print(f"AI: {result}")
            continue
    
    if any(op in user for op in ['+', '-', '*', '/']):
        eq = user.replace(" ", "")
        if '=' not in eq:
            eq = eq + "="
        prompt = eq
        result = calculate(user)
        if result is not None:
            print(f"AI: {result}")
            continue
    else:
        prompt = user + "="
    
    answer = generate(prompt)
    print(f"AI: {answer}")