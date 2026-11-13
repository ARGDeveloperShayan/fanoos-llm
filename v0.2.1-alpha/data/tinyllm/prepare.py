import random
import os
import numpy as np
import pickle
import tiktoken
import json

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

# -------------------- CONFIG --------------------
NUM_EXAMPLES = 100_000
OUT_DIR = "data/fanoosllm"
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ================== DATA GENERATORS (unchanged) ==================
def vary_phrase(options):
    phrase = random.choice(options)
    if random.random() < 0.3: phrase = phrase.capitalize()
    if random.random() < 0.2: phrase = phrase + "?"
    elif random.random() < 0.2: phrase = phrase + "!"
    elif random.random() < 0.3: phrase = phrase + "."
    return phrase

def random_ending():
    return random.choice([
        " What else can I help with?",
        " Feel free to ask another question.",
        " Let me know if you need anything else.",
        " I'm here if you want to explore more.",
        " Does that answer your question?",
        " Pretty cool, right?",
        " Math is fun, isn't it?",
        " Want to try a harder one?",
        " I hope that helps!",
        " Let's keep going!",
        ""
    ])

def math():
    a, b = random.randint(0, 500), random.randint(0, 500)
    op = random.choice(['+', '-', '*', '/'])
    if op == '+': result = a + b
    elif op == '-': result = a - b
    elif op == '*': result = a * b
    else:
        b = max(1, random.randint(1, 500))
        result = a // b
    user = vary_phrase([
        f"What is {a} {op} {b}?",
        f"Calculate {a} {op} {b}.",
        f"How much is {a} {op} {b}?",
        f"Solve {a} {op} {b}.",
        f"{a} {op} {b} = ?"
    ])
    assistant = random.choice([
        f"The answer is {result}.",
        f"That's {result}.",
        f"{result}.",
        f"It's {result}."
    ]) + random_ending()
    return user, assistant

def greeting():
    pairs = [
        ("Hello!", "Hi there! I'm here to help."),
        ("Hi", "Hello! How can I assist you?"),
        ("Hey", "Hey! What can I do for you?"),
        ("Good morning!", "Good morning! Ready to learn?"),
        ("What's up?", "Not much, just waiting to help you!"),
        ("How are you?", "I'm great, thanks for asking!"),
        ("Good afternoon!", "Good afternoon! Let's get started."),
        ("Yo", "Yo! Ask me anything."),
        ("Greetings!", "Greetings! I'm your assistant."),
        ("Hi there", "Hello! I've been looking forward to our chat."),
    ]
    user, assistant = random.choice(pairs)
    return user, assistant + random_ending()

def fact():
    facts = [
        "Honey never spoils.",
        "Octopuses have three hearts.",
        "Bananas are berries, but strawberries aren't.",
        "Your brain generates 12-25 watts of electricity.",
        "A day on Venus is longer than a year on Venus.",
        "Sharks existed before trees.",
        "A group of flamingos is called a 'flamboyance'.",
        "There are more stars in the universe than grains of sand on Earth."
    ]
    user = vary_phrase(["Tell me a fact.", "Give me a cool fact.", "Share something interesting."])
    assistant = random.choice(facts) + random_ending()
    return user, assistant

def banter():
    pairs = [
        ("How are you?", "I'm great, thanks for asking! How about you?"),
        ("Are you smart?", "I like to think so! Want to test me?"),
        ("You're awesome!", "Aw, thanks! You're pretty awesome yourself."),
        ("Do you like math?", "I love it! Math is the language of the universe."),
        ("Tell me a joke.", "Why did the math book look sad? Too many problems!"),
        ("Thanks!", "You're welcome! Happy to help."),
        ("What's your name?", "I'm FanoosLLM, your friendly assistant."),
        ("What can you do?", "I can chat, do math, share facts, and more!"),
    ]
    user, assistant = random.choice(pairs)
    return user, assistant + random_ending()

def why():
    pairs = [
        ("Why is the sky blue?", "Rayleigh scattering causes blue light to scatter more."),
        ("Why do we dream?", "Dreaming helps with memory and emotion processing."),
        ("Why is water wet?", "Water molecules are cohesive, making it feel wet."),
        ("Why do cats purr?", "Purring may help with healing and communication.")
    ]
    user, assistant = random.choice(pairs)
    return user, assistant + random_ending()

# ================== GENERATE ==================
print(f"Generating {NUM_EXAMPLES} conversations...")

data_lines = []
for _ in range(NUM_EXAMPLES):
    r = random.random()
    if r < 0.3:
        user, assistant = math()
    elif r < 0.45:
        user, assistant = greeting()
    elif r < 0.6:
        user, assistant = fact()
    elif r < 0.75:
        user, assistant = banter()
    elif r < 0.9:
        user, assistant = why()
    else:
        user, assistant = math()

    line = f"User: {user}\nAssistant: {assistant}\n"
    data_lines.append(line)

with open(os.path.join(OUT_DIR, "input.txt"), "w", encoding="utf-8") as f:
    f.writelines(data_lines)

print(f"Raw text saved to {OUT_DIR}/input.txt")

# ================== BPE ENCODING ==================
print("Encoding with GPT-2 BPE tokenizer...")

enc = load_gpt2_encoding_offline(
    "data/fanoosllm/gpt2_encoder/encoder.json",
    "data/fanoosllm/gpt2_encoder/vocab.bpe"
)

full_text = "".join(data_lines)
all_ids = enc.encode_ordinary(full_text)
print(f"Total tokens: {len(all_ids)}")

ids_np = np.array(all_ids, dtype=np.uint16)
n = len(ids_np)
val_size = int(0.1 * n)
train_ids = ids_np[:n - val_size]
val_ids = ids_np[n - val_size:]

train_ids.tofile(os.path.join(OUT_DIR, "train.bin"))
val_ids.tofile(os.path.join(OUT_DIR, "val.bin"))

meta = {"vocab_size": 50257}   # GPT-2 vocab size
with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Saved train.bin, val.bin, meta.pkl in {OUT_DIR}/")
print("All done!")