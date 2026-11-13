import random
import os
import numpy as np
import pickle
import tiktoken
import json

# ------------------------------------------------------------------
# Offline BPE loader (same as before)
# ------------------------------------------------------------------
def load_gpt2_encoding_offline(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, 'r', encoding='utf-8') as f:
        encoder = json.load(f)
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

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
NUM_EXAMPLES = 300_000
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def vary_phrase(options):
    phrase = random.choice(options)
    if random.random() < 0.3:
        phrase = phrase.capitalize()
    if random.random() < 0.2:
        phrase = phrase + "?"
    elif random.random() < 0.2:
        phrase = phrase + "!"
    elif random.random() < 0.3:
        phrase = phrase + "."
    return phrase

def random_ending():
    return random.choice([
        " Let me know if you'd like more details.",
        " Feel free to ask anything else!",
        " I'm happy to help further.",
        " Is there anything else you're curious about?",
        " Would you like another fact?",
        " Hope that was interesting!",
        " Do you want to explore another topic?",
        " I can explain more if you'd like.",
        " Let me know what else I can do for you.",
        " I'm here all day!",
        ""
    ])

def long_ending():
    """A longer, friendlier closing."""
    return random.choice([
        " If you have another question, just ask! I'm always ready to chat or help with math, facts, or anything else on your mind.",
        " Feel free to ask follow‑up questions. I enjoy our conversations and am here to make learning fun.",
        " Let me know if you need anything else. I can tell more facts, help with calculations, or just chat.",
        " I hope that answered your question! I'm here whenever you need a quick fact, a calculation, or a friendly chat.",
        ""
    ])

# ------------------------------------------------------------------
# DATA GENERATORS (expanded and with longer answers)
# ------------------------------------------------------------------

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
        f"{a} {op} {b} = ?",
        f"Can you tell me the result of {a} {op} {b}?"
    ])
    assistant = random.choice([
        f"The answer is {result}. I double‑checked my math to be sure.",
        f"That's {result}. I hope that helps with your calculation!",
        f"It's {result}. If you need the steps, I can explain them.",
        f"{result}. Math is one of my favourite subjects!",
        f"I get {result}. Let me know if you have another problem."
    ])
    if random.random() < 0.7:
        assistant += " " + long_ending()
    return user, assistant

def greeting():
    pairs = [
        ("Hello!", "Hi there! I'm so glad you're here. How can I help you today?"),
        ("Hi", "Hello! I'm your AI assistant, ready to answer questions, chat, or tell you something interesting."),
        ("Hey", "Hey! It's great to see you. What can I do for you?"),
        ("Good morning!", "Good morning! I hope your day is off to a wonderful start. What would you like to talk about?"),
        ("What's up?", "Not much, just waiting to help you! Got any questions for me?"),
        ("How are you?", "I'm doing great, thanks for asking! I've been processing a lot of interesting information. How about you?"),
        ("Good afternoon!", "Good afternoon! Ready to learn something new or just chat? I'm here for you."),
        ("Yo", "Yo! Ask me anything – I'm all ears (or bits, actually)."),
        ("Greetings!", "Greetings, friend! I'm your FanoosLLM assistant, full of facts and friendly conversation."),
        ("Hi there", "Hello! I've been looking forward to our chat. What's on your mind?"),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.6:
        assistant += " " + long_ending()
    return user, assistant

def fact():
    facts = [
        "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible!",
        "Octopuses have three hearts. Two pump blood to the gills, while the third pumps it to the rest of the body. Interestingly, the third heart actually stops when the octopus swims.",
        "Bananas are berries, but strawberries aren't. In botanical terms, a berry develops from a single flower with one ovary and has several seeds inside. Strawberries don't meet that definition.",
        "Your brain generates about 12-25 watts of electricity – enough to power a low-wattage LED light bulb. That's pretty impressive for a biological organ!",
        "A day on Venus is longer than a year on Venus. It takes about 243 Earth days to rotate once but only about 225 Earth days to orbit the Sun.",
        "Sharks existed before trees. Sharks first appeared around 400 million years ago, while trees only showed up about 350 million years ago.",
        "A group of flamingos is called a 'flamboyance'. Their bright pink colour comes from the carotenoid pigments in the algae and crustaceans they eat.",
        "There are more stars in the universe than grains of sand on all of Earth's beaches. The observable universe contains roughly 1 sextillion stars."
    ]
    user = vary_phrase(["Tell me a fact.", "Give me a cool fact.", "Share something interesting.", "I want a fun fact!", "What's a mind-blowing fact?"])
    assistant = random.choice(facts)
    if random.random() < 0.5:
        assistant += " " + long_ending()
    return user, assistant

def banter():
    pairs = [
        ("How are you?", "I'm great, thanks for asking! I've been processing a lot of new information and I'm ready to share. How about you?"),
        ("Are you smart?", "I like to think so! I'm constantly learning from the data I was trained on. Want to test me with a tricky question?"),
        ("You're awesome!", "Aw, thank you! You're pretty awesome yourself for taking the time to talk to me. What can I do for you today?"),
        ("Do you like math?", "I absolutely love math! It's the universal language. Whether it's simple addition or something more complex, I'm your calculator buddy."),
        ("Tell me a joke.", "Why did the math book look sad? Because it had too many problems! I've got more where that came from if you need a laugh."),
        ("Thanks!", "You're very welcome! I'm happy to help. Is there anything else you'd like?"),
        ("What's your name?", "I'm FanoosLLM, your friendly pocket assistant. I may be small, but I'm full of knowledge!"),
        ("What can you do?", "I can chat, answer questions, tell facts, help with math, and much more. Just ask me anything!"),
        ("Do you ever get tired?", "Nope! I don't need sleep like humans do. I'm always ready whenever you need me."),
        ("What's the meaning of life?", "That's a big question! Different people have different answers. For me, the meaning of life is to help and learn as much as possible. What do you think?"),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.5:
        assistant += " " + long_ending()
    return user, assistant

def why():
    pairs = [
        ("Why is the sky blue?", "The sky appears blue because of Rayleigh scattering. Sunlight is made up of all colours, but blue light is scattered more in the Earth's atmosphere, so we see a blue sky most of the time."),
        ("Why do we dream?", "Scientists believe dreaming helps with memory consolidation, emotional processing, and creative problem‑solving. Dreams may also be the brain's way of rehearsing responses to future events."),
        ("Why is water wet?", "Water feels wet because its molecules are polar and stick to each other (cohesion). When you touch water, these cohesive forces create surface tension, which gives the sensation of wetness."),
        ("Why do cats purr?", "Cats purr not only when they're content but also when they're stressed or healing. The vibrations, around 25-150 Hz, may promote bone growth and tissue repair."),
        ("Why does the moon have phases?", "The phases are caused by the relative positions of the Earth, Moon, and Sun. We see different portions of the moon's sunlit side depending on where it is in its orbit."),
        ("Why do leaves change colour in autumn?", "As days get shorter and temperatures drop, chlorophyll breaks down, revealing yellow and orange pigments that were present all along. Some trees produce red pigments as well."),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.4:
        assistant += " " + long_ending()
    return user, assistant

def general_knowledge():
    pairs = [
        ("What is the capital of France?", "The capital of France is Paris. It's often called the 'City of Light' and is famous for landmarks like the Eiffel Tower and the Louvre Museum."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system. It's a gas giant with a mass one‑thousandth that of the Sun, and it has a Great Red Spot that is a giant storm."),
        ("How many continents are there?", "There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America. Some models combine Europe and Asia into Eurasia."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare, the famous English playwright, around 1595. It's one of the most performed plays in history."),
        ("What is the speed of light?", "The speed of light in a vacuum is about 299,792,458 metres per second, often rounded to 300,000 km/s. It's the fastest speed possible in the universe."),
        ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain above sea level, at 8,848.86 metres. However, from base to summit, Mauna Kea in Hawaii is taller when measured from the ocean floor."),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.6:
        assistant += " " + long_ending()
    return user, assistant

def definition():
    pairs = [
        ("What is photosynthesis?", "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesise foods from carbon dioxide and water. It produces oxygen as a by‑product."),
        ("What is gravity?", "Gravity is a fundamental force of nature that attracts two bodies with mass. It's what keeps us on the ground and causes planets to orbit the Sun."),
        ("What is DNA?", "DNA, or deoxyribonucleic acid, is the molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all known organisms."),
        ("What is artificial intelligence?", "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. It includes learning, reasoning, and self‑correction."),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.5:
        assistant += " " + long_ending()
    return user, assistant

def simple_reasoning():
    pairs = [
        ("If I have 3 apples and give away 2, how many do I have left?", "You would have 1 apple left. It's simple subtraction: 3 minus 2 equals 1."),
        ("What comes next in the sequence: 2, 4, 6, 8, ...?", "The next number is 10. The pattern adds 2 each time."),
        ("If a pizza is cut into 8 slices and I eat 3, how many slices are left?", "There would be 5 slices left. 8 minus 3 equals 5."),
        ("True or false: all birds can fly.", "False! Not all birds can fly. For example, ostriches, penguins, and kiwis are birds that cannot fly."),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.5:
        assistant += " " + long_ending()
    return user, assistant

def personal_questions():
    pairs = [
        ("Can you tell me a story?", "Once upon a time, in a land made entirely of numbers, a little zero felt lonely... But that's a story for another time! What kind of story would you like?"),
        ("I'm feeling sad.", "I'm sorry to hear that. Sometimes talking about it helps. I'm here to listen or distract you with facts and jokes."),
        ("Goodbye!", "Bye! Come back anytime you need help or just want to say hi. Take care!"),
        ("Thank you!", "You're welcome! It's my pleasure to assist. Don't hesitate to ask if you need anything else."),
    ]
    user, assistant = random.choice(pairs)
    if random.random() < 0.4:
        assistant += " " + long_ending()
    return user, assistant

# ------------------------------------------------------------------
# GENERATE
# ------------------------------------------------------------------
print(f"Generating {NUM_EXAMPLES} conversations...")
data_lines = []
for _ in range(NUM_EXAMPLES):
    r = random.random()
    if r < 0.20:
        user, assistant = math()
    elif r < 0.30:
        user, assistant = greeting()
    elif r < 0.40:
        user, assistant = fact()
    elif r < 0.50:
        user, assistant = banter()
    elif r < 0.60:
        user, assistant = why()
    elif r < 0.70:
        user, assistant = general_knowledge()
    elif r < 0.78:
        user, assistant = definition()
    elif r < 0.85:
        user, assistant = simple_reasoning()
    elif r < 0.92:
        user, assistant = personal_questions()
    else:
        user, assistant = math()
    line = f"User: {user}\nAssistant: {assistant}\n"
    data_lines.append(line)

with open(os.path.join(OUT_DIR, "input.txt"), "w", encoding="utf-8") as f:
    f.writelines(data_lines)
print(f"Raw text saved to {OUT_DIR}/input.txt")

# ------------------------------------------------------------------
# BPE ENCODING (offline)
# ------------------------------------------------------------------
print("Encoding with GPT-2 BPE tokenizer...")
enc = load_gpt2_encoding_offline(
    "data/gpt2_encoder/encoder.json",
    "data/gpt2_encoder/vocab.bpe"
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
meta = {"vocab_size": 50257}
with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"Saved train.bin, val.bin, meta.pkl in {OUT_DIR}/")
print("All done!")