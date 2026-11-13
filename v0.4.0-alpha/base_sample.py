import torch
from fanoos import FanoosConfig, FanoosLLM, load_tokenizer, clean_state_dict
tok, _ = load_tokenizer()
ck = torch.load("checkpoints_pretrain/best.pt", map_location="cuda", weights_only=False)
m = FanoosLLM(FanoosConfig(**ck["cfg"])).cuda()
m.load_state_dict(clean_state_dict(ck["model"]), strict=False); m.eval()
for p in ["The capital of France is", "Photosynthesis is the process by which", "The Roman Empire"]:
    x = torch.tensor([tok.encode(p, add_special_tokens=False)]).cuda()
    print(tok.decode(m.generate(x, max_new_tokens=60, temperature=0.7, top_k=50, top_p=0.9)[0]), "\n---")