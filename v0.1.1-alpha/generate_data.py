import random
import os
from config import NUM_EXAMPLES, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

CONVERSATIONS = [
    # Simple greetings
    ("hello", "hello! how can i help you?"),
    ("hi", "hi there! 😊"),
    ("hey", "hey! nice to see you"),
    ("good morning", "good morning! ready to chat?"),
    ("good afternoon", "good afternoon! what's up?"),
    ("howdy", "howdy partner! 🤠"),
    ("what's up", "not much, just chatting with you!"),
    ("greetings", "greetings, human!"),
    ("yo", "yo! what's happening?"),
    ("hi there", "hi! tell me something interesting."),

    # Talking algorithms / casual conversation
    ("how are you", "i'm doing great, thanks for asking!"),
    ("how are you today", "i'm fantastic! and you?"),
    ("what's your name", "i'm your friendly chat assistant!"),
    ("who are you", "i'm a small talk bot, nice to meet you!"),
    ("tell me a joke", "why do programmers prefer dark mode? less light attracts fewer bugs! 😄"),
    ("how's the weather", "it's always sunny in the digital world! ☀️"),
    ("what can you do", "i can chat, answer simple questions, and keep you company."),
    ("thank you", "you're very welcome!"),
    ("thanks", "anytime! 😊"),
    ("bye", "goodbye! see you later."),
    ("goodbye", "take care! come back soon."),
    ("see you later", "see you! have a great day."),
    ("nice to meet you", "likewise! i enjoy our chat."),
    ("what do you like", "i like learning new words and talking with people like you!"),
    ("are you a robot", "yep! but a friendly one."),
]

def random_conversation():
    """Return a string in the format 'prompt=answer|' with total length ≤ 128."""
    prompt, answer = random.choice(CONVERSATIONS)
    line = f"{prompt}={answer}|"
    if len(line) > 128:
        max_answer_len = 128 - len(prompt) - 2
        answer = answer[:max_answer_len]
        line = f"{prompt}={answer}|"
    return line

data_path = os.path.join(MODEL_DIR, "knowledge.txt")
print(f"Generating {NUM_EXAMPLES} conversational examples...")
with open(data_path, "w", encoding="utf-8") as f:
    for i in range(NUM_EXAMPLES):
        line = random_conversation()
        f.write(line + "\n")
        if i % 200000 == 0 and i > 0:
            print(f"  {i} examples done...")
print(f"Data file created: {data_path}")