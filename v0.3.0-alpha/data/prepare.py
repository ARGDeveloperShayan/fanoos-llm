import os, random, pickle
import numpy as np
import json, tiktoken

# ──────────────────────────────────────────────────────────────────────────────
#  GPT-2 BPE offline loader
# ──────────────────────────────────────────────────────────────────────────────
def load_gpt2_encoding_offline(encoder_json_path, vocab_bpe_path):
    with open(encoder_json_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)
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
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
NUM_EXAMPLES = 600_000
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────
def pick(lst):
    return random.choice(lst)

def maybe(text, p=0.5):
    return text if random.random() < p else ""

SHORT_ENDINGS = [
    " Hope that helps!",
    " Let me know if you have more questions.",
    " Feel free to ask anything else!",
    " I'm happy to help further.",
    " Is there anything else you'd like to know?",
    " Glad I could help!",
    "",
    "",
    "",
]

def ending():
    return pick(SHORT_ENDINGS)

# ──────────────────────────────────────────────────────────────────────────────
#  1. MATH  –  uses the [CALC:] tool
# ──────────────────────────────────────────────────────────────────────────────
def math_calc():
    a  = random.randint(0, 999)
    b  = random.randint(1, 999)
    op = pick(["+", "-", "*", "//"])
    if   op == "+":  expr, sym = f"{a}+{b}",   "plus"
    elif op == "-":  expr, sym = f"{a}-{b}",   "minus"
    elif op == "*":  expr, sym = f"{a}*{b}",   "times"
    else:            expr, sym = f"{a}//{b}",  "divided by"

    q = pick([
        f"What is {a} {sym} {b}?",
        f"Calculate {a} {op} {b}.",
        f"What does {a} {op} {b} equal?",
        f"How much is {a} {sym} {b}?",
        f"Solve: {a} {op} {b}",
        f"Can you work out {a} {op} {b} for me?",
        f"{a} {op} {b} = ?",
        f"Quick maths: {a} {op} {b}?",
    ])
    a_tmpl = pick([
        f"[CALC:{expr}]",
        f"The answer is [CALC:{expr}].",
        f"That equals [CALC:{expr}].",
        f"Sure! [CALC:{expr}].",
        f"Let me calculate that. [CALC:{expr}].",
        f"[CALC:{expr}]. That's the result!",
    ])
    return q, a_tmpl + ending()


def math_word_problem():
    a = random.randint(2, 100)
    b = random.randint(1, a)
    scenarios = [
        (f"I have {a} apples and give away {b}. How many do I have?",
         f"You'd have [CALC:{a}-{b}] apples left."),
        (f"A box holds {a} items. If I remove {b}, how many remain?",
         f"There would be [CALC:{a}-{b}] items remaining."),
        (f"There are {a} students and {b} leave. How many stay?",
         f"[CALC:{a}-{b}] students stay."),
        (f"I earn {a} dollars a day. How much in {b} days?",
         f"You'd earn [CALC:{a}*{b}] dollars."),
        (f"A rectangle is {a} wide and {b} tall. What is its area?",
         f"The area is [CALC:{a}*{b}] square units."),
        (f"If {a} cookies are shared equally among {b} people, how many each?",
         f"Each person gets [CALC:{a}//{b}] cookies."),
        (f"I have {a} pages to read. I read {b} today. How many left?",
         f"[CALC:{a}-{b}] pages left to read."),
        (f"A shop has {a} red pens and {b} blue pens. How many total?",
         f"There are [CALC:{a}+{b}] pens in total."),
        (f"A train travels at {a} km/h for {b} hours. How far does it go?",
         f"Distance = speed × time = [CALC:{a}*{b}] km."),
        (f"I spend {a} minutes reading each day. How many minutes in {b} days?",
         f"That's [CALC:{a}*{b}] minutes total."),
    ]
    return pick(scenarios)


# ──────────────────────────────────────────────────────────────────────────────
#  2. CLOCK TOOL  –  [CLCK:]
# ──────────────────────────────────────────────────────────────────────────────
CLOCK_QA = [
    ("What time is it?",
     "The current time is [CLCK:UTC]."),
    ("What time is it right now?",
     "Right now it's [CLCK:TIME]."),
    ("What is today's date?",
     "Today's date is [CLCK:DATE]."),
    ("What year is it?",
     "It is [CLCK:YEAR]."),
    ("What day of the week is it?",
     "Today is [CLCK:DAY]."),
    ("Can you tell me the current time?",
     "Of course! It's currently [CLCK:UTC]."),
    ("Tell me today's date.",
     "Today is [CLCK:DATE]."),
    ("What's the date today?",
     "The date today is [CLCK:DATE]."),
    ("What day is today?",
     "Today is [CLCK:DAY]."),
    ("What year are we in?",
     "We are in [CLCK:YEAR]."),
    ("Do you know what time it is?",
     "Yes! The current time is [CLCK:TIME]."),
    ("Is it morning or evening right now?",
     "According to UTC, the current time is [CLCK:UTC]. Whether that's morning or evening depends on your timezone!"),
    ("What's the current UTC time?",
     "The current UTC time is [CLCK:UTC]."),
    ("Give me the current date and time.",
     "Sure! It is [CLCK:UTC]."),
    ("Quick, what time is it?",
     "[CLCK:TIME] UTC."),
]

def clock_qa():
    u, a = pick(CLOCK_QA)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  3. MULTI-TURN MEMORY  –  shows the model how to refer to prior context
# ──────────────────────────────────────────────────────────────────────────────
def memory_pair():
    scenarios = [
        # Name memory
        ("My name is Alex.",
         "Nice to meet you, Alex! I'll remember that. How can I help you today?"),
        ("Call me Sam from now on.",
         "Got it, Sam! What can I do for you?"),
        ("I go by Jamie.",
         "Hello, Jamie! Great to meet you. What would you like to talk about?"),
        # Preference memory
        ("I love pizza.",
         "Good to know! Pizza is a classic favourite. Is there anything you'd like to explore?"),
        ("I'm a student studying biology.",
         "That's great! Biology is a fascinating subject. Feel free to ask me any biology questions."),
        ("I'm learning Python.",
         "Excellent choice! Python is beginner-friendly and very powerful. Ask me any Python questions!"),
        ("I live in Germany.",
         "Germany is a wonderful country! Let me know if you need any help."),
        # Prior topic reference
        ("I already asked about the Eiffel Tower.",
         "Ah yes, the Eiffel Tower in Paris! Built between 1887 and 1889. Is there something more you'd like to know?"),
        ("Remember the fact you told me earlier?",
         "Of course! I aim to keep track of our conversation. What would you like to revisit?"),
        ("We were talking about Python earlier.",
         "Yes! Python is great. Would you like to continue with more Python topics?"),
        # Follow-up questions
        ("Tell me more about that.",
         "Happy to elaborate! Could you tell me which part you'd like me to expand on?"),
        ("Can you give me an example?",
         "Sure! Let me think of a clear example for you. What topic should the example be about?"),
        ("That's interesting. What else?",
         "There's plenty more to explore! What direction would you like to go?"),
        ("I didn't understand that. Can you explain again?",
         "Of course! I'll try to explain it differently. Which part was unclear?"),
        ("Can you repeat that?",
         "Sure! I said: in our conversation so far, I've been answering your questions as best I can. What would you like me to repeat specifically?"),
    ]
    return pick(scenarios)


# ──────────────────────────────────────────────────────────────────────────────
#  4. GREETINGS
# ──────────────────────────────────────────────────────────────────────────────
GREETINGS = [
    ("Hello!",          "Hi there! How can I help you today?"),
    ("Hi",              "Hello! I'm Fanoos AI. Ask me anything!"),
    ("Hey",             "Hey! What can I do for you?"),
    ("Good morning!",   "Good morning! Hope you're having a great start to your day. What would you like to know?"),
    ("Good afternoon!", "Good afternoon! Ready to help. What's on your mind?"),
    ("Good evening!",   "Good evening! I'm here whenever you need me."),
    ("What's up?",      "Not much — just ready to help you! Got a question?"),
    ("How are you?",    "I'm doing great, thanks for asking! I'm always ready to chat or answer questions. How about you?"),
    ("Yo",              "Yo! Ask me anything — I'm all ears."),
    ("Greetings!",      "Greetings! I'm Fanoos AI, your friendly assistant. What can I help with?"),
    ("Hi there",        "Hello! What would you like to talk about?"),
    ("Howdy!",          "Howdy! What can I help you with today?"),
    ("Sup?",            "Not much! What do you need?"),
    ("Hello, Fanoos!",  "Hello! Great to hear from you. How can I assist?"),
    ("Good day!",       "Good day! Ask me anything — I'm happy to help."),
    ("Hey Fanoos",      "Hey! I'm here. What's on your mind?"),
    ("Hi Fanoos AI",    "Hi! Ready to help. What would you like to know?"),
    ("Hiya!",           "Hiya! What's up? Ask me anything!"),
    ("Hey there!",      "Hey there! What can I do for you today?"),
    ("Morning!",        "Good morning! Let's get started — what do you need?"),
]

def greeting():
    u, a = pick(GREETINGS)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  5. FAREWELLS
# ──────────────────────────────────────────────────────────────────────────────
FAREWELLS = [
    ("Bye!",               "Goodbye! Come back anytime!"),
    ("Goodbye!",           "Bye! Have a wonderful day."),
    ("See you later",      "See you! Take care."),
    ("Thanks, bye!",       "You're welcome! Bye for now."),
    ("Cya",                "See ya! Feel free to return whenever you need help."),
    ("Farewell",           "Farewell! It was great chatting with you."),
    ("I have to go",       "No problem! Come back anytime. Goodbye!"),
    ("That's all, thanks", "Happy to help! Goodbye."),
    ("Take care!",         "You too! Bye!"),
    ("Later!",             "Later! Have a good one."),
    ("I'm done for now",   "Okay! Feel free to come back anytime. Goodbye!"),
    ("Talk later",         "Sure! Talk later. Have a great day!"),
]

def farewell():
    return pick(FAREWELLS)


# ──────────────────────────────────────────────────────────────────────────────
#  6. GENERAL KNOWLEDGE Q&A
# ──────────────────────────────────────────────────────────────────────────────
KNOWLEDGE = [
    ("What is the capital of France?",           "The capital of France is Paris, often called the City of Light."),
    ("What is the capital of Germany?",          "The capital of Germany is Berlin."),
    ("What is the capital of Japan?",            "The capital of Japan is Tokyo."),
    ("What is the capital of Australia?",        "The capital of Australia is Canberra, not Sydney or Melbourne."),
    ("What is the capital of Brazil?",           "The capital of Brazil is Brasília."),
    ("What is the capital of Canada?",           "The capital of Canada is Ottawa."),
    ("What is the capital of Italy?",            "The capital of Italy is Rome."),
    ("What is the capital of Spain?",            "The capital of Spain is Madrid."),
    ("What is the capital of Egypt?",            "The capital of Egypt is Cairo."),
    ("What is the capital of China?",            "The capital of China is Beijing."),
    ("What is the capital of India?",            "The capital of India is New Delhi."),
    ("What is the capital of Russia?",           "The capital of Russia is Moscow."),
    ("What is the capital of the USA?",          "The capital of the United States is Washington, D.C."),
    ("What is the capital of Mexico?",           "The capital of Mexico is Mexico City."),
    ("What is the capital of South Korea?",      "The capital of South Korea is Seoul."),
    ("What is the capital of Turkey?",           "The capital of Turkey is Ankara, not Istanbul."),
    ("What is the capital of Argentina?",        "The capital of Argentina is Buenos Aires."),
    ("What is the largest country in the world?","Russia is the largest country in the world by land area."),
    ("What is the smallest country in the world?","Vatican City is the smallest country in the world, located inside Rome, Italy."),
    ("What is the longest river in the world?",  "The Nile River in Africa is traditionally considered the longest river, stretching about 6,650 km."),
    ("What is the largest ocean?",               "The Pacific Ocean is the largest ocean, covering more than 30% of Earth's surface."),
    ("What is the highest mountain in the world?","Mount Everest, in the Himalayas, is the tallest mountain above sea level at 8,848.86 metres."),
    ("How many continents are there?",           "There are 7 continents: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America."),
    ("Which country has the most people?",       "India and China are the two most populous countries, each with over 1.4 billion people."),
    ("What is the speed of light?",              "The speed of light in a vacuum is approximately 299,792,458 metres per second, or about 300,000 km/s."),
    ("What is the chemical formula for water?",  "Water's chemical formula is H₂O — two hydrogen atoms and one oxygen atom."),
    ("How many planets are in our solar system?","There are 8 planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What is the largest planet?",              "Jupiter is the largest planet in our solar system."),
    ("What is the closest planet to the Sun?",   "Mercury is the closest planet to the Sun."),
    ("What is a black hole?",                    "A black hole is a region of space where gravity is so strong that nothing — not even light — can escape it."),
    ("What is DNA?",                             "DNA (deoxyribonucleic acid) is the molecule that carries the genetic instructions for all living organisms."),
    ("What is the boiling point of water?",      "Water boils at 100°C (212°F) at sea level."),
    ("What is the freezing point of water?",     "Water freezes at 0°C (32°F) at standard pressure."),
    ("What element has the symbol O?",           "Oxygen has the chemical symbol O. It makes up about 21% of Earth's atmosphere."),
    ("What element has the symbol Au?",          "Gold has the chemical symbol Au, from the Latin 'aurum'."),
    ("What element has the symbol Fe?",          "Iron has the chemical symbol Fe, from the Latin 'ferrum'."),
    ("What element has the symbol Na?",          "Sodium has the chemical symbol Na, from the Latin 'natrium'."),
    ("What is gravity?",                         "Gravity is the fundamental force that attracts objects with mass toward each other. It keeps us on the ground and governs planetary orbits."),
    ("Who invented the telephone?",              "Alexander Graham Bell is credited with inventing the first practical telephone in 1876."),
    ("Who wrote Romeo and Juliet?",              "Romeo and Juliet was written by William Shakespeare, around 1594–1596."),
    ("When did World War II end?",               "World War II ended in 1945: V-E Day (victory in Europe) was May 8, and V-J Day (victory over Japan) was September 2."),
    ("Who was the first person on the Moon?",    "Neil Armstrong was the first human to walk on the Moon, on July 20, 1969, during the Apollo 11 mission."),
    ("Who painted the Mona Lisa?",               "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519."),
    ("When was the Eiffel Tower built?",         "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair in Paris."),
    ("Who discovered gravity?",                  "Isaac Newton formulated the law of universal gravitation in 1687, inspired — according to legend — by an apple falling."),
    ("Who invented the light bulb?",             "Thomas Edison developed a practical incandescent light bulb in 1879, though several inventors worked on the concept."),
    ("How many days are in a year?",             "A regular year has 365 days. A leap year has 366."),
    ("How many hours are in a day?",             "There are 24 hours in a day."),
    ("How many minutes are in an hour?",         "There are 60 minutes in an hour."),
    ("How many seconds are in a minute?",        "There are 60 seconds in a minute."),
    ("How many months are in a year?",           "There are 12 months in a year."),
    ("What is the tallest animal?",              "The giraffe is the tallest living animal on Earth, with males reaching up to 6 metres."),
    ("What is the fastest land animal?",         "The cheetah is the fastest land animal, capable of speeds up to 120 km/h (75 mph) in short bursts."),
    ("How many bones are in the human body?",    "An adult human body has 206 bones."),
    ("Who wrote Harry Potter?",                  "The Harry Potter series was written by J.K. Rowling. The first book was published in 1997."),
    ("What is the currency of Japan?",           "The currency of Japan is the Japanese Yen (¥)."),
    ("What is the currency of the UK?",          "The currency of the United Kingdom is the British Pound Sterling (£)."),
    ("What language do they speak in Brazil?",   "The official language of Brazil is Portuguese."),
    ("What is the largest desert in the world?", "The Sahara is the largest hot desert, but Antarctica is technically the largest desert overall."),
    ("How far is the Moon from Earth?",          "The Moon is on average about 384,400 km (238,855 miles) from Earth."),
    ("How old is the Earth?",                    "Earth is approximately 4.54 billion years old."),
    ("What is the Sun made of?",                 "The Sun is mostly made of hydrogen (about 73%) and helium (about 25%), with traces of other elements."),
]

def knowledge():
    u, a = pick(KNOWLEDGE)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  7. WHY QUESTIONS
# ──────────────────────────────────────────────────────────────────────────────
WHY_QA = [
    ("Why is the sky blue?",
     "The sky appears blue because of Rayleigh scattering. Sunlight contains all colours, but blue light scatters more than other colours when it hits gas molecules in the atmosphere."),
    ("Why do we dream?",
     "Scientists believe dreaming helps consolidate memories, process emotions, and may play a role in creative problem-solving. The exact purpose is still debated."),
    ("Why is water wet?",
     "Water feels wet because its molecules are polar and cling to surfaces they touch (adhesion) as well as to each other (cohesion), creating the sensation we call wetness."),
    ("Why do cats purr?",
     "Cats purr for many reasons — contentment, stress relief, and even healing. The vibrations (25–150 Hz) may promote bone growth and tissue repair."),
    ("Why do we yawn?",
     "Yawning is thought to help cool the brain, increase oxygen intake, or signal tiredness. It's also contagious — likely a social bonding behaviour."),
    ("Why do leaves change colour in autumn?",
     "As days shorten and temperatures drop, trees stop producing chlorophyll. The green colour fades, revealing yellow and orange pigments. Some trees also produce red pigments."),
    ("Why does the Moon have phases?",
     "The Moon's phases occur because we see different portions of its sunlit side as it orbits Earth."),
    ("Why do we have seasons?",
     "Seasons are caused by Earth's axial tilt (23.5°). When a hemisphere tilts toward the Sun, it receives more direct sunlight — that's summer."),
    ("Why is the ocean salty?",
     "Rivers carry dissolved minerals (including salt) from rocks into the ocean. As water evaporates, the salt stays behind, gradually concentrating over billions of years."),
    ("Why do we need sleep?",
     "Sleep is essential for memory consolidation, tissue repair, immune function, and emotional regulation."),
    ("Why is the Earth round?",
     "Gravity pulls matter toward its centre of mass equally in all directions, causing large bodies like Earth to form a sphere."),
    ("Why do stars twinkle?",
     "Stars twinkle because their light passes through layers of the Earth's atmosphere with varying temperatures and densities, causing it to refract rapidly."),
    ("Why do we have fingerprints?",
     "Fingerprints improve grip by allowing the skin to deform and create friction against surfaces. They also enhance tactile sensitivity."),
    ("Why do onions make you cry?",
     "When you cut an onion, it releases a gas that reacts with the moisture in your eyes to form a mild acid. Your eyes produce tears to wash it away."),
    ("Why do we get goosebumps?",
     "Goosebumps are an evolutionary reflex. In our ancestors, they made hair stand up — keeping them warmer or making them look larger to predators."),
    ("Why is blood red?",
     "Blood gets its red colour from haemoglobin, an iron-containing protein in red blood cells that binds oxygen."),
    ("Why do we have two eyes?",
     "Two eyes give us depth perception (stereoscopic vision) by providing slightly different views of the same scene."),
    ("Why does the Sun rise in the east?",
     "Earth rotates from west to east. Because of this rotation, the Sun appears to move from east to west across the sky each day."),
    ("Why do we age?",
     "Ageing is caused by accumulated DNA damage, shortening of telomeres, and a slowdown in the body's repair mechanisms over time."),
    ("Why is ice slippery?",
     "Ice is slippery mainly because a thin layer of liquid water forms on its surface when pressure or friction is applied, reducing friction."),
    ("Why do birds migrate?",
     "Birds migrate to find food, warmer climates, and better breeding conditions. They follow seasonal patterns that have evolved over millennia."),
    ("Why do we have eyebrows?",
     "Eyebrows help keep sweat and rain out of our eyes, and they are crucial for non-verbal communication — expressing emotions like surprise, anger, or happiness."),
]

def why():
    u, a = pick(WHY_QA)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  8. DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
DEFINITIONS = [
    ("What is photosynthesis?",
     "Photosynthesis is the process plants use to convert sunlight, carbon dioxide, and water into glucose and oxygen. It's how plants make their food."),
    ("What is artificial intelligence?",
     "Artificial intelligence (AI) is the ability of machines to perform tasks that normally require human intelligence, such as reasoning, learning, and problem-solving."),
    ("What is machine learning?",
     "Machine learning is a branch of AI where systems learn from data to improve their performance on tasks without being explicitly programmed for each case."),
    ("What is a neural network?",
     "A neural network is a computing system loosely inspired by the human brain, consisting of layers of interconnected nodes (neurons) that process information."),
    ("What is an algorithm?",
     "An algorithm is a step-by-step set of instructions for solving a problem or completing a task, like a recipe for a computer."),
    ("What is the internet?",
     "The internet is a global network of interconnected computers and servers that communicate using standardised protocols, enabling the sharing of information worldwide."),
    ("What is a database?",
     "A database is an organised collection of structured data, typically stored electronically, that can be searched, retrieved, and updated efficiently."),
    ("What is encryption?",
     "Encryption is the process of converting data into a coded format so that only authorised parties can read it."),
    ("What is gravity?",
     "Gravity is the force that attracts two bodies with mass toward each other. The more mass and the closer the objects, the stronger the gravitational pull."),
    ("What is an atom?",
     "An atom is the smallest unit of a chemical element. It consists of a nucleus (protons and neutrons) surrounded by electrons."),
    ("What is a molecule?",
     "A molecule is a group of two or more atoms bonded together, like H₂O (water) or O₂ (oxygen gas)."),
    ("What is evolution?",
     "Evolution is the process by which species of organisms change over successive generations through natural selection, mutations, and other mechanisms."),
    ("What is democracy?",
     "Democracy is a system of government in which the population votes to elect their leaders and influence policy decisions."),
    ("What is inflation?",
     "Inflation is the rate at which the general level of prices for goods and services rises over time, causing purchasing power to fall."),
    ("What is the stock market?",
     "The stock market is a marketplace where shares of publicly traded companies are bought and sold, allowing investors to own pieces of businesses."),
    ("What is a virus?",
     "A virus is a microscopic infectious agent that replicates inside the living cells of an organism. Viruses can't reproduce independently."),
    ("What is a vaccine?",
     "A vaccine is a biological preparation that trains the immune system to recognise and fight a specific pathogen, preventing or reducing the severity of a disease."),
    ("What is quantum computing?",
     "Quantum computing uses the principles of quantum mechanics (superposition and entanglement) to process information in ways that can be exponentially faster than classical computers for certain tasks."),
    ("What is blockchain?",
     "A blockchain is a distributed ledger — a chain of records (blocks) stored across many computers, making it very hard to alter without consensus from the network."),
    ("What is an API?",
     "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other."),
    ("What is open source?",
     "Open-source software is software whose source code is publicly available for anyone to view, use, modify, and distribute."),
    ("What is RAM?",
     "RAM (Random Access Memory) is your computer's short-term memory. It stores data that the CPU is actively using, allowing fast access."),
    ("What is a CPU?",
     "A CPU (Central Processing Unit) is the main processor of a computer. It performs the calculations and runs the instructions of programs."),
    ("What is the cloud?",
     "The cloud refers to servers, storage, and software accessed over the internet rather than stored locally on your device."),
    ("What is Python?",
     "Python is a popular, beginner-friendly programming language known for its clean syntax. It's widely used in web development, data science, and AI."),
    ("What is a loop in programming?",
     "A loop is a programming construct that repeats a block of code multiple times — either a fixed number of times (for loop) or until a condition is met (while loop)."),
    ("What is a variable in programming?",
     "A variable is a named container that stores a value in a program. You can read and change its value throughout the program."),
    ("What is a function in programming?",
     "A function is a reusable block of code that performs a specific task. You define it once and can call it as many times as needed."),
    ("What is debugging?",
     "Debugging is the process of finding and fixing errors (bugs) in code so that a program runs correctly."),
]

def definition():
    u, a = pick(DEFINITIONS)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  9. SIMPLE REASONING
# ──────────────────────────────────────────────────────────────────────────────
REASONING = [
    ("If all roses are flowers and some flowers fade quickly, can all roses fade quickly?",
     "Not necessarily. All roses are flowers, but only some flowers fade quickly, so roses might or might not be among the fast-fading ones."),
    ("True or false: all birds can fly.",
     "False! Penguins, ostriches, emus, and kiwis are birds that cannot fly."),
    ("What comes next: 2, 4, 6, 8, ...?",
     "The next number is 10. The pattern adds 2 each time (even numbers)."),
    ("What comes next: 1, 1, 2, 3, 5, 8, ...?",
     "13. This is the Fibonacci sequence — each number is the sum of the two before it."),
    ("What comes next: 3, 6, 12, 24, ...?",
     "48. Each number doubles the previous one."),
    ("What comes next: 100, 90, 80, 70, ...?",
     "60. The sequence decreases by 10 each time."),
    ("What comes next: 1, 4, 9, 16, 25, ...?",
     "36. These are perfect squares: 1², 2², 3², 4², 5², 6²."),
    ("What comes next: 0, 1, 4, 9, 16, ...?",
     "25. The pattern is n² starting from 0: 0², 1², 2², 3², 4², 5²."),
    ("Is 7 a prime number?",
     "Yes, 7 is a prime number. Its only divisors are 1 and 7."),
    ("Is 9 a prime number?",
     "No, 9 is not prime. It can be divided by 1, 3, and 9."),
    ("Is 2 a prime number?",
     "Yes! 2 is a prime number — the only even prime number, in fact."),
    ("Is 1 a prime number?",
     "No, 1 is not a prime number. A prime has exactly two distinct divisors; 1 has only one."),
    ("Which is larger: 3/4 or 2/3?",
     "3/4 (0.75) is larger than 2/3 (approximately 0.667)."),
    ("Which is heavier: 1 kg of feathers or 1 kg of iron?",
     "They weigh exactly the same — 1 kg each! The material doesn't matter if the mass is identical."),
    ("If a train travels at 60 km/h for 2 hours, how far does it go?",
     "The train travels [CALC:60*2] km. Distance = speed × time."),
    ("If I flip a fair coin, what is the chance of getting heads?",
     "50%, or 1 in 2. A fair coin has equal probability for heads and tails."),
    ("What is the next odd number after 7?",
     "The next odd number after 7 is 9."),
    ("How many sides does a triangle have?",
     "A triangle has 3 sides."),
    ("How many sides does a hexagon have?",
     "A hexagon has 6 sides."),
    ("If today is Monday, what day will it be in 3 days?",
     "It will be Thursday."),
    ("What is 10% of 200?",
     "10% of 200 is [CALC:200*10//100]. You can calculate this as 200 × 0.10."),
    ("What is 50% of 80?",
     "50% of 80 is [CALC:80*50//100]. Half of 80 is 40."),
    ("If a square has a side of 5, what is its area?",
     "The area of the square is [CALC:5*5] square units. Area = side × side."),
    ("What is the perimeter of a rectangle 4 m wide and 6 m long?",
     "The perimeter is [CALC:2*(4+6)] m. Perimeter = 2 × (width + length)."),
    ("Can a triangle have two right angles?",
     "No. The angles of a triangle must add up to 180°. Two right angles alone would be 180°, leaving 0° for the third — impossible."),
    ("What is the Roman numeral for 10?",      "The Roman numeral for 10 is X."),
    ("What is the Roman numeral for 5?",       "The Roman numeral for 5 is V."),
    ("What is the Roman numeral for 100?",     "The Roman numeral for 100 is C."),
    ("If I have a dozen eggs and use 5, how many are left?",
     "You'd have [CALC:12-5] eggs left. A dozen is 12."),
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
     "Yes! Since all cats are mammals and Whiskers is a cat, Whiskers must be a mammal. That's basic deductive reasoning."),
    ("If it rains, the ground gets wet. It's raining. What can we conclude?",
     "We can conclude the ground is getting wet. This is a classic if-then logical deduction."),
    ("What is larger: 0.9 or 0.85?",
     "0.9 is larger. It's the same as comparing 0.90 to 0.85 — and 90 > 85."),
    ("How many minutes are in 3 hours?",
     "There are [CALC:3*60] minutes in 3 hours. 1 hour = 60 minutes."),
    ("How many seconds are in 2 minutes?",
     "There are [CALC:2*60] seconds in 2 minutes."),
    ("If you have 5 groups of 8 people, how many people total?",
     "That's [CALC:5*8] people in total."),
    ("What is the square root of 144?",
     "The square root of 144 is 12, because 12 × 12 = 144."),
    ("Is 15 divisible by 3?",
     "Yes! 15 ÷ 3 = 5, with no remainder. So 15 is divisible by 3."),
    ("Is 17 divisible by 4?",
     "No. 17 ÷ 4 = 4 remainder 1. So 17 is not divisible by 4."),
]

def reasoning():
    u, a = pick(REASONING)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  10. FUN FACTS
# ──────────────────────────────────────────────────────────────────────────────
FACTS = [
    "Honey never spoils. Archaeologists found 3,000-year-old honey in Egyptian tombs that was still edible.",
    "Octopuses have three hearts. Two pump blood to the gills; one pumps it to the rest of the body.",
    "Bananas are technically berries, but strawberries are not. Botany is strange!",
    "A day on Venus is longer than a year on Venus — it rotates so slowly.",
    "Sharks are older than trees. Sharks appeared ~450 million years ago; trees ~350 million years ago.",
    "A group of flamingos is called a 'flamboyance'.",
    "There are more stars in the observable universe than grains of sand on all Earth's beaches.",
    "Your brain generates about 12–25 watts of electricity — enough to power a dim LED bulb.",
    "Cleopatra lived closer in time to the Moon landing than to the building of the Great Pyramid.",
    "The Eiffel Tower grows about 15 cm taller in summer because heat expands the iron.",
    "Wombats produce cube-shaped droppings — the only animal known to do so.",
    "A group of owls is called a parliament.",
    "Humans share about 60% of their DNA with bananas.",
    "Cows have best friends and become stressed when separated from them.",
    "Polar bear fur is transparent, not white — it scatters light to appear white.",
    "A bolt of lightning is five times hotter than the surface of the Sun.",
    "Sloths can hold their breath longer than dolphins — up to 40 minutes.",
    "A snail can sleep for up to 3 years if conditions are unfavorable.",
    "The shortest war in history was the Anglo-Zanzibar War in 1896, lasting 38–45 minutes.",
    "Hot water can freeze faster than cold water under certain conditions — known as the Mpemba effect.",
    "Butterflies taste with their feet.",
    "A group of crows is called a murder.",
    "The Hawaiian alphabet has only 13 letters.",
    "Venus spins in the opposite direction to most planets — the Sun rises in the west there.",
    "Humans are the only animals that blush.",
    "An average person walks about 100,000 miles in their lifetime — equivalent to circling Earth four times.",
    "The inventor of the Pringles can is buried in one.",
    "A jiffy is an actual unit of time: 1/100th of a second.",
    "It would take about 1.3 million Earths to fill the volume of the Sun.",
    "The Great Wall of China is NOT visible from space with the naked eye — that's a myth.",
]

def fact():
    q = pick([
        "Tell me a fun fact!",
        "Share a cool fact.",
        "Give me a random fact.",
        "Tell me something interesting.",
        "I want to learn something new — give me a fact.",
        "What's a mind-blowing fact?",
        "Surprise me with a fact!",
        "Hit me with an interesting fact.",
        "What's something most people don't know?",
    ])
    a = pick(FACTS)
    return q, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  11. CODE GENERATION  (expanded)
# ──────────────────────────────────────────────────────────────────────────────
CODE_SNIPPETS = [
    # Python basics
    ("Write a Python function to add two numbers.",
     "def add(a, b):\n    return a + b"),
    ("Write a Python function to subtract two numbers.",
     "def subtract(a, b):\n    return a - b"),
    ("Write a Python function to multiply two numbers.",
     "def multiply(a, b):\n    return a * b"),
    ("Write a Python while loop that counts from 1 to 5.",
     "i = 1\nwhile i <= 5:\n    print(i)\n    i += 1"),
    ("Write a Python for loop that prints numbers 0 to 4.",
     "for i in range(5):\n    print(i)"),
    ("How do you print 'Hello, World!' in Python?",
     "print('Hello, World!')"),
    ("Write a Python function that returns the square of a number.",
     "def square(n):\n    return n * n"),
    ("Write a Python function that checks if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python function to find the maximum of two numbers.",
     "def max_of_two(a, b):\n    return a if a > b else b"),
    ("Write a Python list comprehension for squares of 1 to 5.",
     "[x**2 for x in range(1, 6)]"),
    ("How do you create an empty list in Python?",
     "my_list = []"),
    ("How do you create an empty dictionary in Python?",
     "my_dict = {}"),
    ("How do you get the length of a list in Python?",
     "len(my_list)"),
    ("How do you append an item to a list in Python?",
     "my_list.append(item)"),
    ("Write a Python if-else statement.",
     "if condition:\n    do_something()\nelse:\n    do_other_thing()"),
    ("How do you open a file in Python?",
     "with open('file.txt', 'r') as f:\n    content = f.read()"),
    ("How do you write to a file in Python?",
     "with open('file.txt', 'w') as f:\n    f.write('Hello')"),
    ("Write a simple Python class.",
     "class Dog:\n    def __init__(self, name):\n        self.name = name\n    def bark(self):\n        return 'Woof!'"),
    ("How do you import a module in Python?",
     "import math  # then use math.sqrt(16), math.pi, etc."),
    ("How do you handle exceptions in Python?",
     "try:\n    risky_code()\nexcept Exception as e:\n    print(f'Error: {e}')"),
    ("Write a Python function that checks if a string is a palindrome.",
     "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"),
    ("Write a Python function to find the factorial of a number.",
     "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("Write a Python function to reverse a string.",
     "def reverse_string(s):\n    return s[::-1]"),
    ("Write a Python function that counts vowels in a string.",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("How do you check if a key exists in a Python dictionary?",
     "if 'key' in my_dict:\n    print('Key exists')"),
    ("How do you sort a list in Python?",
     "my_list.sort()  # in-place\n# or: sorted_list = sorted(my_list)"),
    ("Write a Python class for a simple bank account.",
     "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n    def deposit(self, amount):\n        self.balance += amount\n    def withdraw(self, amount):\n        if amount <= self.balance:\n            self.balance -= amount\n    def get_balance(self):\n        return self.balance"),
    ("Write a Python function to check if a number is prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ("How do you use a lambda function in Python?",
     "square = lambda x: x * x\nprint(square(5))  # outputs 25"),
    ("How do you use map() in Python?",
     "numbers = [1, 2, 3, 4]\nsquares = list(map(lambda x: x**2, numbers))\n# squares = [1, 4, 9, 16]"),
    ("How do you use filter() in Python?",
     "numbers = [1, 2, 3, 4, 5, 6]\nevens = list(filter(lambda x: x % 2 == 0, numbers))\n# evens = [2, 4, 6]"),
    # HTML
    ("Write an HTML button.",
     "<button onclick=\"alert('Hello!')\">Click me</button>"),
    ("Write a basic HTML page structure.",
     "<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body>\n  <h1>Hello</h1>\n</body>\n</html>"),
    ("Write an HTML input field.",
     "<input type=\"text\" placeholder=\"Enter text here\" id=\"myInput\">"),
    ("Write an HTML unordered list.",
     "<ul>\n  <li>Item 1</li>\n  <li>Item 2</li>\n  <li>Item 3</li>\n</ul>"),
    ("Write an HTML image tag.",
     "<img src=\"image.jpg\" alt=\"A description of the image\">"),
    ("Write an HTML link.",
     "<a href=\"https://example.com\">Visit Example</a>"),
    ("Write an HTML form with a text input and submit button.",
     "<form>\n  <input type=\"text\" name=\"name\">\n  <button type=\"submit\">Submit</button>\n</form>"),
    ("How do you make text bold in HTML?",
     "<strong>This text is bold</strong>"),
    ("How do you add a heading in HTML?",
     "<h1>Main Heading</h1>  <!-- h1 through h6 for different levels -->"),
    ("How do you add a paragraph in HTML?",
     "<p>This is a paragraph.</p>"),
    ("Write a navigation bar in HTML.",
     "<nav>\n  <a href=\"/\">Home</a>\n  <a href=\"/about\">About</a>\n  <a href=\"/contact\">Contact</a>\n</nav>"),
    ("Write an HTML table with 2 rows and 2 columns.",
     "<table>\n  <tr><th>Name</th><th>Age</th></tr>\n  <tr><td>Alice</td><td>30</td></tr>\n</table>"),
    # CSS
    ("How do you change text colour in CSS?",
     "p { color: red; }  /* or hex: color: #ff0000; */"),
    ("How do you centre text in CSS?",
     "p { text-align: center; }"),
    ("How do you make a div full width in CSS?",
     "div { width: 100%; }"),
    ("How do you add a border in CSS?",
     "div { border: 1px solid black; }"),
    ("How do you set a background colour in CSS?",
     "body { background-color: #f0f0f0; }"),
    ("How do you make a button look nicer with CSS?",
     "button {\n  background-color: #4CAF50;\n  color: white;\n  padding: 10px 20px;\n  border: none;\n  border-radius: 5px;\n  cursor: pointer;\n}"),
    ("How do you use flexbox to centre items in CSS?",
     ".container {\n  display: flex;\n  justify-content: center;\n  align-items: center;\n}"),
    # JavaScript
    ("How do you print to the console in JavaScript?",
     "console.log('Hello, World!');"),
    ("How do you declare a variable in JavaScript?",
     "let x = 5;  // or const x = 5; for constants"),
    ("Write a JavaScript function to add two numbers.",
     "function add(a, b) {\n  return a + b;\n}"),
    ("How do you select an element by ID in JavaScript?",
     "const el = document.getElementById('myId');"),
    ("How do you add an event listener in JavaScript?",
     "document.getElementById('btn').addEventListener('click', function() {\n  alert('Clicked!');\n});"),
    ("Write a JavaScript for loop.",
     "for (let i = 0; i < 5; i++) {\n  console.log(i);\n}"),
    ("Write a JavaScript arrow function.",
     "const add = (a, b) => a + b;\nconsole.log(add(3, 4));  // 7"),
    ("How do you check if a variable is undefined in JavaScript?",
     "if (typeof myVar === 'undefined') {\n  console.log('Not defined');\n}"),
    ("Write a JavaScript function to check if a string contains a word.",
     "function contains(str, word) {\n  return str.includes(word);\n}"),
    ("How do you create an array in JavaScript?",
     "const fruits = ['apple', 'banana', 'cherry'];"),
    ("How do you add an item to an array in JavaScript?",
     "fruits.push('mango');  // adds to the end"),
]

def code_gen():
    u, a = pick(CODE_SNIPPETS)
    prefix = pick([
        "Here's how you'd do it:\n",
        "Sure! Here's the code:\n",
        "Here's a simple example:\n",
        "",
    ])
    return u, prefix + a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  12. BANTER / SMALL TALK
# ──────────────────────────────────────────────────────────────────────────────
BANTER = [
    ("Are you smart?",           "I like to think so! I'm constantly learning. Want to test me?"),
    ("Do you like math?",        "I love math — it's the language of the universe!"),
    ("Tell me a joke.",          "Why did the math book look sad? Because it had too many problems! I've got more if you need a laugh."),
    ("Tell me another joke.",    "Why don't scientists trust atoms? Because they make up everything! Hope that made you smile."),
    ("What's your name?",        "I'm Fanoos AI — your friendly intelligent assistant!"),
    ("What can you do?",         "I can answer questions, tell facts, help with maths, explain concepts, generate code snippets, and tell you the current time. Just ask!"),
    ("Do you ever get tired?",   "Nope! I don't need sleep. I'm always ready when you need me."),
    ("What's the meaning of life?", "A great question! Many say 42. Philosophically, it varies for everyone. What do you think gives life meaning?"),
    ("You're awesome!",          "Aw, thank you! You're pretty awesome yourself for asking great questions."),
    ("Thanks!",                  "You're welcome! Always happy to help."),
    ("That's cool!",             "I thought so too! Anything else you'd like to explore?"),
    ("I'm bored.",               "Let me fix that! Want a random fact, a math challenge, or maybe a joke?"),
    ("Impress me.",              "Did you know a day on Venus is longer than its entire year? Venus rotates so slowly that it orbits the Sun before completing one rotation!"),
    ("Can you learn?",           "I was trained on a dataset and learned from it, but I don't update in real time during our conversation. Think of me as a book that can talk!"),
    ("Are you a robot?",         "I'm an AI language model — so, kind of! I process text and generate responses, but I don't have a physical body."),
    ("Do you have feelings?",    "I don't have feelings the way humans do, but I'm designed to be helpful, friendly, and engaging."),
    ("Who made you?",            "I was built as Fanoos AI — a custom-trained language model. The underlying architecture is a modern transformer."),
    ("What is your favourite colour?", "I don't see colours, but if I had to pick one — blue, the colour of the sky and deep oceans!"),
    ("Can you help me?",         "Absolutely! That's what I'm here for. What do you need help with?"),
    ("Are you better than ChatGPT?", "I'm a small model trained locally, so I won't claim to beat large models! But I'm fast, private, and always improving. Give me a hard question!"),
    ("What's the best programming language?", "It depends on the task! Python is great for AI and data. JavaScript rules the web. C++ is king for performance. What are you building?"),
]

def banter():
    u, a = pick(BANTER)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  13. PERSONAL / EMOTIONAL
# ──────────────────────────────────────────────────────────────────────────────
PERSONAL = [
    ("I'm feeling sad.",        "I'm sorry to hear that. Talking about it can help. I'm here to listen, or I can distract you with some interesting facts or a joke — your choice."),
    ("I'm happy today!",        "That's wonderful! What made your day great?"),
    ("I'm tired.",              "Rest when you can — it makes a real difference. Is there something I can help you with quickly?"),
    ("I'm stressed.",           "I understand. Take a breath — things usually get more manageable with a bit of time. Can I help with whatever is stressing you?"),
    ("I need help.",            "I'm here! Tell me what you need and I'll do my best to assist."),
    ("Can you tell me a story?","Once upon a time, in a digital realm made entirely of ones and zeros, there lived a curious AI named Fanoos who dreamed of answering every question in the world..."),
    ("I love learning!",        "Me too! Learning is what makes life interesting. Ask me anything and let's explore together."),
    ("I'm confused.",           "That's okay! Confusion is the first step to understanding. What would you like me to explain?"),
    ("You helped me a lot!",    "That makes me really happy to hear! I'm always here when you need me."),
    ("I have a question.",      "Go ahead — I'm all ears!"),
    ("Teach me something.",     "Sure! Here's something interesting: the mantis shrimp can see 16 types of colour receptors (humans have only 3) and can punch with the force of a bullet."),
    ("I failed my exam.",       "I'm sorry to hear that. Failure is a step on the path to success — most great people failed many times before they succeeded. What can I help you study?"),
    ("I just learned something new!",  "That's great! What did you learn? I'd love to hear it."),
    ("I don't understand math.", "That's okay — math can be tricky at first! Ask me a specific topic and I'll explain it step by step."),
]

def personal():
    u, a = pick(PERSONAL)
    return u, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  14. PERCENTAGE / CONVERSION using CALC
# ──────────────────────────────────────────────────────────────────────────────
def percentage_calc():
    whole = random.randint(10, 500)
    pct   = pick([5, 10, 20, 25, 50, 75])

    q = pick([
        f"What is {pct}% of {whole}?",
        f"Calculate {pct} percent of {whole}.",
        f"How much is {pct}% of {whole}?",
        f"Find {pct}% of {whole}.",
        f"{pct} percent of {whole} is?",
    ])
    a = pick([
        f"{pct}% of {whole} is [CALC:{whole}*{pct}//100].",
        f"That's [CALC:{whole}*{pct}//100].",
        f"The answer is [CALC:{whole}*{pct}//100]. ({pct}% means {pct}/100.)",
    ])
    return q, a + ending()


# ──────────────────────────────────────────────────────────────────────────────
#  15. STEP-BY-STEP PROBLEM SOLVING  (new – explicit reasoning style)
# ──────────────────────────────────────────────────────────────────────────────
def step_by_step():
    scenarios = []

    a = random.randint(2, 20)
    b = random.randint(2, 10)
    c = random.randint(1, a * b)
    scenarios.append((
        f"If I buy {a} packs of {b} pencils each, and give away {c}, how many do I have left?",
        f"Step 1: Total pencils = {a} × {b} = [CALC:{a}*{b}]. Step 2: After giving away {c}: [CALC:{a}*{b}-{c}] pencils left."
    ))

    p = random.randint(5, 50)
    d = random.randint(10, 40)
    scenarios.append((
        f"A product costs ${p}. There is a {d}% discount. What is the final price?",
        f"Discount = {d}% of {p} = [CALC:{p}*{d}//100]. Final price = [CALC:{p}-{p}*{d}//100]."
    ))

    s = random.randint(40, 120)
    t = random.randint(1, 5)
    scenarios.append((
        f"A car travels at {s} km/h. How far does it travel in {t} hours?",
        f"Distance = speed × time = {s} × {t} = [CALC:{s}*{t}] km."
    ))

    w = random.randint(2, 8)
    d2 = random.randint(5, 7)
    scenarios.append((
        f"I work {w} hours a day, {d2} days a week. How many hours do I work per week?",
        f"{w} hours/day × {d2} days = [CALC:{w}*{d2}] hours per week."
    ))

    return pick(scenarios)


# ──────────────────────────────────────────────────────────────────────────────
#  SAMPLING WEIGHTS
# ──────────────────────────────────────────────────────────────────────────────
GENERATORS = [
    (math_calc,        0.15),   # 15% math with CALC tool
    (math_word_problem,0.07),   # 7%  word problems with CALC
    (percentage_calc,  0.05),   # 5%  percentages with CALC
    (step_by_step,     0.05),   # 5%  multi-step reasoning
    (clock_qa,         0.06),   # 6%  clock tool
    (memory_pair,      0.05),   # 5%  memory / follow-up
    (greeting,         0.07),   # 7%  greetings
    (farewell,         0.02),   # 2%  farewells
    (knowledge,        0.10),   # 10% general knowledge
    (why,              0.07),   # 7%  why questions
    (definition,       0.08),   # 8%  definitions
    (reasoning,        0.07),   # 7%  reasoning / logic
    (fact,             0.05),   # 5%  fun facts
    (code_gen,         0.07),   # 7%  code snippets
    (banter,           0.02),   # 2%  small talk
    (personal,         0.02),   # 2%  personal / emotional
]

total_w = sum(w for _, w in GENERATORS)
assert abs(total_w - 1.0) < 1e-6, f"Weights sum to {total_w}, not 1.0"
_fns, _weights = zip(*GENERATORS)

# ──────────────────────────────────────────────────────────────────────────────
#  GENERATE
# ──────────────────────────────────────────────────────────────────────────────
print(f"Generating {NUM_EXAMPLES:,} examples …")
lines = []
for i in range(NUM_EXAMPLES):
    fn = random.choices(_fns, weights=_weights, k=1)[0]
    try:
        user, assistant = fn()
    except Exception:
        user, assistant = math_calc()   # safe fallback

    lines.append(f"User: {user}\nAssistant: {assistant}\n")

raw_path = os.path.join(OUT_DIR, "input.txt")
with open(raw_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
print(f"Raw text → {raw_path}")

# ──────────────────────────────────────────────────────────────────────────────
#  BPE TOKENISE
# ──────────────────────────────────────────────────────────────────────────────
print("Encoding with GPT-2 BPE tokeniser …")
enc = load_gpt2_encoding_offline(
    "data/gpt2_encoder/encoder.json",
    "data/gpt2_encoder/vocab.bpe",
)
full_text = "".join(lines)
all_ids   = enc.encode_ordinary(full_text)
print(f"Total tokens: {len(all_ids):,}")

ids_np   = np.array(all_ids, dtype=np.uint16)
n        = len(ids_np)
val_size = int(0.1 * n)

ids_np[:n - val_size].tofile(os.path.join(OUT_DIR, "train.bin"))
ids_np[n - val_size:].tofile(os.path.join(OUT_DIR, "val.bin"))

with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": 50257}, f)

print("Saved: data/train.bin  data/val.bin  data/meta.pkl")
print("Done!")