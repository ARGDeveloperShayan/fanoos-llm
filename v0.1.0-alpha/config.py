"""
Configuration file for TinyLLM project.
Contains all hyperparameters and configuration settings.
"""

# Vocabulary settings
VOCAB_SIZE = None  # Will be set dynamically based on vocab.py

# Model architecture hyperparameters
EMBEDDING_DIM = 512        # Larger embedding dimension for richer representations
NUM_HEADS = 8              # More heads for finer-grained attention patterns
NUM_LAYERS = 8             # Deeper decoder stack
FEEDFORWARD_DIM = 1536     # Wider feedforward for capacity
DROPOUT = 0.1              # Dropout probability

# Sequence settings
MAX_SEQ_LEN = 96           # Maximum sequence length (including BOS/EOS tokens)

# Training hyperparameters
BATCH_SIZE = 32            # Batch size for training
LEARNING_RATE = 0.0003     # Slightly lower LR for larger model stability
NUM_EPOCHS = 80            # More steps for convergence
GRAD_CLIP = 1.0            # Gradient clipping threshold

# Data generation settings
TRAIN_DATA_SIZE = 60000    # Number of training examples to generate
VAL_DATA_SIZE = 6000       # Number of validation examples to generate

# Generation settings
MAX_GENERATION_LEN = 96    # Maximum tokens to generate during inference
GENERATION_TEMPERATURE = 0.7  # Cooler sampling for more stable answers
TOP_K = 8                  # Top-k sampling (0 to disable)

# Device settings
# Prefer CUDA; fallback to CPU if not available (checked in train/generate).
DEVICE = 'cuda'

# Mixed precision (speeds up on RTX GPUs)
USE_AMP = True

# File paths
MODEL_SAVE_PATH = 'model.pt'
VOCAB_SAVE_PATH = 'vocab.pkl'

