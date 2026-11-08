from __future__ import annotations
import argparse
import asyncio
import os
import time
import uuid
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from fanoos import DEFAULT_SYSTEM_PROMPT, build_prompt, load_tokenizer
from fanoos import FanoosConfig
from fanoos import FanoosLLM

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Fanoos AI")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATE: dict = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "fanoos-ai"
    messages: List[Message]
    stream: bool = False
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 512

@torch.no_grad()
def generate_tokens(prompt_ids: list[int], req: ChatRequest):
    model, cfg, device = STATE["model"], STATE["cfg"], STATE["device"]
    im_end = STATE["special"]["im_end"]

    ids = torch.tensor([prompt_ids], device=device)
    logits, _, cache = model(ids, use_cache=True)
    for _ in range(req.max_tokens):
        next_tok = model._sample(logits[:, -1, :], req.temperature, req.top_k, req.top_p)
        tok_id = int(next_tok[0, 0])
        if tok_id == im_end:
            break
        yield tok_id
        logits, _, cache = model(next_tok, past_key_values=cache, use_cache=True)

def sse_chunk(content: str, finish: Optional[str] = None) -> str:
    import json

    delta = {"content": content} if content else {}
    if finish:
        delta = {}
    chunk = {
        "id": STATE.get("request_id", ""),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fanoos-ai",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk)}\n\n"

@app.get("/")
def index():
    return FileResponse(os.path.join(PROJECT_ROOT, "static", "index.html"))

@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": "fanoos-ai", "object": "model", "owned_by": "fanoos"}]}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    tok, special = STATE["tok"], STATE["special"]
    STATE["request_id"] = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    messages = [m.model_dump() for m in req.messages]
    prompt = build_prompt(messages, system_prompt=DEFAULT_SYSTEM_PROMPT, add_generation_prompt=True)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)

    if req.stream:
        async def event_stream():
            yield sse_chunk("")
            for tok_id in generate_tokens(prompt_ids, req):
                text = tok.decode([tok_id])
                await asyncio.sleep(0)
                yield sse_chunk(text)
            yield sse_chunk("", finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text = "".join(tok.decode([t]) for t in generate_tokens(prompt_ids, req))
    return {
        "id": STATE["request_id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "fanoos-ai",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text.strip()},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": 0, "total_tokens": 0},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.checkpoint
    if ckpt is None:
        for name in ("best.pt", "final.pt", "ckpt.pt"):
            p = os.path.join(PROJECT_ROOT, "checkpoints", name)
            if os.path.isfile(p):
                ckpt = p
                break
    if ckpt is None or not os.path.isfile(ckpt):
        raise SystemExit("No checkpoint found. Train first (python train.py) or pass --checkpoint.")

    tok, special = load_tokenizer()
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = FanoosConfig(**payload["cfg"])
    model = FanoosLLM(cfg).to(device)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.eval()

    STATE.update(model=model, tok=tok, special=special, cfg=cfg, device=device)
    print(f"[serve] {model.num_parameters() / 1e6:.1f}M params on {device} | {ckpt}")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()