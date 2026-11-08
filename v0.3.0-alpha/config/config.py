# config/fanoos.py  –  FanoosLLM v1 configuration
# ──────────────────────────────────────────────────────────────
# Architecture: 8 layers × 8 heads × 512 dim  ≈ 117 M parameters
# Fits comfortably in 8 GB VRAM; trains on RTX 5060 in hours.
#
# To use a smaller/faster model (≈ 25 M params):
#   n_layer = 6 ; n_head = 6 ; n_embd = 384
#
# To use a larger model (≈ 117 M params, GPT-2 size):
#   n_layer = 12 ; n_head = 12 ; n_embd = 768
# ──────────────────────────────────────────────────────────────

# I/O
out_dir   = "model"
eval_interval = 500
eval_iters    = 40
log_interval  = 10
always_save_checkpoint = False
wandb_log = False

# Data
dataset  = ""          # prepare.py writes directly to data/
gradient_accumulation_steps = 4
batch_size  = 16
block_size  = 512      # context window (tokens)

# Model  – modern settings
n_layer     = 8
n_head      = 8
n_embd      = 512
dropout     = 0.1
bias        = False    # no bias on linear layers (handled in model.py)
rope_theta  = 10000.0  # RoPE base frequency

# Optimiser
learning_rate   = 3e-4
max_iters       = 30000
lr_decay_iters  = 30000
min_lr          = 3e-5
beta1           = 0.9
beta2           = 0.99
weight_decay    = 1e-1
grad_clip       = 1.0
warmup_iters    = 300

# System  (torch must be imported in the script that exec()s this file)
compile = False        # set True if you have PyTorch 2.0+ and want torch.compile