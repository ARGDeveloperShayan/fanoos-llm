# config/tinyllm.py

out_dir = 'tinyllm'
eval_interval = 500
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'tinyllm'
gradient_accumulation_steps = 4
batch_size = 16
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

learning_rate = 3e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 3e-5
beta2 = 0.99
warmup_iters = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile = False