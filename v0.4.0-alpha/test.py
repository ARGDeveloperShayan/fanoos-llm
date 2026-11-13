from __future__ import annotations

import argparse
import os
import torch

from fanoos import (
    FanoosConfig,
    FanoosLLM,
    build_prompt,
    decode_tokens,
    load_tokenizer,
    DEFAULT_SYSTEM_PROMPT,
    clean_state_dict,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TEST_SUITE: dict[str, dict] = {
    "1. Basic conversation": {
        "type": "single",
        "questions": [
            "Hello!",
            "Hi, how are you?",
            "What's your name?",
            "Who are you?",
            "What can you do?",
            "Are you an AI model?",
            "Who created you?",
            "Nice to meet you!",
            "Thank you for your help.",
            "Goodbye!",
        ],
    },
    "2. General knowledge": {
        "type": "single",
        "questions": [
            "What is the capital of France?",
            "How many planets are in our solar system?",
            "Who invented the telephone?",
            "What is the largest ocean on Earth?",
            "Why is the sky blue?",
            "What is the difference between a star and a planet?",
            "How does the internet work?",
            "What is artificial intelligence?",
            "Why do humans need sleep?",
            "What is the speed of light?",
        ],
    },
    "3. Reasoning and logic": {
        "type": "single",
        "questions": [
            "If I have 5 apples and give away 2, how many apples do I have left?",
            "A farmer has 17 sheep. All but 9 run away. How many remain?",
            "If today is Wednesday, what day will it be in 10 days?",
            "A bat and a ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
            "If all cats are animals and some animals are black, does that mean some cats are black? Explain.",
            "What comes next in this sequence: 2, 4, 8, 16, 32, ...?",
            "You are in a race and overtake the person in second place. What position are you in?",
            "A doctor gives you three pills and tells you to take one every 30 minutes. How long will it take to finish all three?",
        ],
    },
    "4. Mathematics": {
        "type": "single",
        "questions": [
            "What is 125 x 48?",
            "Calculate 15% of 240.",
            "Solve: 3x + 7 = 22.",
            "What is the square root of 144?",
            "If a car travels at 80 km/h for 2.5 hours, how far does it travel?",
            "Calculate: 2^8 + 3^3.",
            "What is the probability of rolling a 6 with a fair six-sided die?",
            "A product costs $80 after a 20% discount. What was its original price?",
        ],
    },
    "5. Language and multilingual": {
        "type": "single",
        "questions": [
            "Explain what machine learning is in simple words.",
            "What is the difference between AI and human intelligence?",
            "هوش مصنوعی چیست؟ به زبان ساده توضیح بده.",
            "تفاوت یادگیری ماشین و یادگیری عمیق چیست؟",
            "چرا آسمان آبی است؟",
            "Was ist künstliche Intelligenz?",
            "Erkläre den Unterschied zwischen Wetter und Klima.",
            "Qu'est-ce que l'intelligence artificielle?",
            "Explique-moi comment fonctionne Internet.",
            "¿Qué es el aprendizaje automático?",
            "¿Por qué es importante el agua para la vida?",
            'Translate "Artificial intelligence is changing the world" into Persian.',
            'Translate "من در حال ساخت یک مدل هوش مصنوعی هستم" into English.',
            'Translate "Guten Morgen, wie geht es dir?" into Spanish.',
        ],
    },
    "6. Coding abilities": {
        "type": "single",
        "questions": [
            "Write a Python function that adds two numbers.",
            "Explain what a variable is in programming.",
            "Write a Python program that checks whether a number is even or odd.",
            "What is the difference between JavaScript and Python?",
            "Find the error in this code:\n\ndef add(a, b):    return a - bprint(add(5, 3))",
            "Write a simple HTML page with a heading and a button.",
            "Explain what a loop does.",
        ],
    },
    "7. Creativity": {
        "type": "single",
        "questions": [
            "Write a short story about a robot that wants to become human.",
            "Invent a new technology that could change the world.",
            "Write a funny joke about programmers.",
            "Imagine what the world will look like in 100 years.",
            "Create a name for a futuristic AI company.",
            "Write a four-line poem about space.",
        ],
    },
    "8. Instruction following": {
        "type": "single",
        "questions": [
            "Explain gravity in exactly 10 words.",
            "Give me three reasons why exercise is important. No more, no less.",
            "Answer using only one word: Is the Earth round?",
            'Explain photosynthesis without using the word "plant".',
            'Write a paragraph without using the letter "e".',
            "Give me five fruits in alphabetical order.",
            "Explain quantum computing as if I were five years old.",
        ],
    },
    "9. Context and memory": {
        "type": "conversation",
        "messages": [
            "My name is Alex, I am 19 years old, and my favorite color is blue.",
            "I am currently learning Python and building a small game.",
            "What is my name?",
            "What programming language am I learning?",
            "What is my favorite color?",
        ],
    },
    "10. Stress testing": {
        "type": "single",
        "questions": [
            "Explain the theory of relativity in detail.",
            "Compare nuclear energy and solar energy.",
            "Give me a 20-step plan to build a successful technology startup.",
            "Explain how a Transformer neural network works, including attention.",
            "Write a 500-word article about the future of artificial intelligence.",
            "Explain a complicated scientific concept using three different examples.",
            "If you don't know the answer to a question, what should you do?",
        ],
    },
}

@torch.no_grad()
def generate(
    model: FanoosLLM,
    tok,
    special: dict,
    messages: list[dict],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    prompt = build_prompt(messages, add_generation_prompt=True, system_prompt=DEFAULT_SYSTEM_PROMPT)
    ids = tok.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], device=device)
    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=special["im_end"],
    )
    reply_ids = out[0, len(ids):].tolist()
    return decode_tokens(tok, reply_ids)

def find_checkpoint(explicit: str | None) -> str:
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"checkpoint not found: {explicit}")
        return explicit
    for name in ("best.pt", "final.pt", "ckpt.pt"):
        p = os.path.join(PROJECT_ROOT, "checkpoints", name)
        if os.path.isfile(p):
            return p
    raise SystemExit(
        "No checkpoint found in checkpoints/. "
        "Train first (python train.py ...) or pass --checkpoint."
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0.0 = greedy/deterministic; 0.7 is a good default")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--category", default=None,
                    help="run only one category (prefix match, e.g. '3')")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N questions per category")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = find_checkpoint(args.checkpoint)
    tok, special = load_tokenizer()

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = FanoosConfig(**payload["cfg"])
    cfg.resolve_vocab_size(len(tok))

    model = FanoosLLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(clean_state_dict(payload["model"]), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.eval()

    print(f"[test] checkpoint : {ckpt_path}")
    print(f"[test] iter       : {payload.get('iter', '?')}  "
          f"| best_val: {payload.get('best_val', '?')}")
    print(f"[test] params     : {model.num_parameters() / 1e6:.2f}M")
    print(f"[test] device     : {device}")
    print(f"[test] sampling   : T={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} max_new={args.max_new_tokens}")

    selected = TEST_SUITE
    if args.category:
        selected = {k: v for k, v in TEST_SUITE.items() if k.startswith(args.category)}
        if not selected:
            raise SystemExit(f"no category matched '{args.category}'")

    for cat_name, spec in selected.items():
        print("\n" + "=" * 72)
        print(cat_name)
        print("=" * 72)

        if spec["type"] == "single":
            questions = spec["questions"]
            if args.limit:
                questions = questions[: args.limit]
            for i, q in enumerate(questions, 1):
                print(f"\n[{i}] user: {q}")
                reply = generate(
                    model, tok, special,
                    [{"role": "user", "content": q}],
                    device, args.max_new_tokens,
                    args.temperature, args.top_k, args.top_p,
                )
                print(f"[{i}] llm:  {reply}")

        else:
            history: list[dict] = []
            messages = spec["messages"]
            if args.limit:
                messages = messages[: args.limit]
            for i, msg in enumerate(messages, 1):
                print(f"\n[{i}] user: {msg}")
                history.append({"role": "user", "content": msg})
                reply = generate(
                    model, tok, special, history,
                    device, args.max_new_tokens,
                    args.temperature, args.top_k, args.top_p,
                )
                history.append({"role": "assistant", "content": reply})
                print(f"[{i}] llm:  {reply}")

    print("\n[test] done.")

if __name__ == "__main__":
    main()