"""
Text generation module for FanoosLLM.
Implements auto-regressive generation with greedy decoding.
"""

import torch
import torch.nn.functional as F
import config
from vocab import Vocabulary
from model import FanoosLLMModel


def generate_text(prompt: str, model: FanoosLLMModel, vocab: Vocabulary,
                  max_length: int = None, temperature: float = None, top_k: int = None) -> str:
    """
    Generate text from a prompt using the model.
    
    Args:
        prompt: Input text prompt
        model: Trained FanoosLLM model
        vocab: Vocabulary instance
        max_length: Maximum generation length (uses config if None)
        temperature: Sampling temperature (uses config if None)
    
    Returns:
        Generated text string
    """
    # Set defaults
    max_length = max_length or config.MAX_GENERATION_LEN
    temperature = temperature or config.GENERATION_TEMPERATURE
    top_k = top_k if top_k is not None else getattr(config, "TOP_K", 0)
    
    # Set device
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()
    
    # Encode prompt
    input_ids = vocab.encode(prompt, add_bos=True, add_eos=False)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate tokens auto-regressively
    generated_ids = input_ids.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Current sequence length
            current_len = len(generated_ids)
            
            # Truncate if too long
            if current_len >= config.MAX_SEQ_LEN:
                break
            
            # Prepare input (last sequence)
            input_seq = torch.tensor([generated_ids], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(input_seq)  # (1, seq_len, vocab_size)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Sample with temperature + top-k
            next_token_id = _sample_next_token(next_token_logits, temperature, top_k)
            
            # Check for EOS token
            if next_token_id == vocab.eos_id:
                break
            
            # Append generated token
            generated_ids.append(next_token_id)
    
    # Decode generated sequence
    # Remove the original prompt tokens (keep only newly generated tokens)
    # Actually, let's return the full sequence for now
    generated_text = vocab.decode(generated_ids, skip_special=True)
    
    return generated_text


def generate_response(prompt: str, model: FanoosLLMModel, vocab: Vocabulary,
                     max_length: int = None, temperature: float = None, top_k: int = None) -> str:
    """
    Generate a response to a prompt (returns only the generated part, not the prompt).
    
    Args:
        prompt: Input text prompt
        model: Trained FanoosLLM model
        vocab: Vocabulary instance
        max_length: Maximum generation length
        temperature: Sampling temperature
    
    Returns:
        Generated response text (without the prompt)
    """
    # Set defaults
    max_length = max_length or config.MAX_GENERATION_LEN
    temperature = temperature or config.GENERATION_TEMPERATURE
    top_k = top_k if top_k is not None else getattr(config, "TOP_K", 0)
    
    # Set device
    device = torch.device(config.DEVICE)
    model = model.to(device)
    model.eval()
    
    # Encode prompt (with BOS)
    prompt_ids = vocab.encode(prompt, add_bos=True, add_eos=False)
    prompt_len = len(prompt_ids)
    
    # Start with prompt tokens
    generated_ids = prompt_ids.copy()
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        for _ in range(max_length):
            # Current sequence length
            current_len = len(generated_ids)
            
            # Truncate if too long
            if current_len >= config.MAX_SEQ_LEN:
                break
            
            # Prepare input
            input_seq = torch.tensor([generated_ids], dtype=torch.long).to(device)
            
            # Forward pass
            logits = model(input_seq)  # (1, seq_len, vocab_size)
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            next_token_id = _sample_next_token(next_token_logits, temperature, top_k)
            
            # Check for EOS token
            if next_token_id == vocab.eos_id:
                break
            
            # Append generated token
            generated_ids.append(next_token_id)
    
    # Extract only the generated part (after prompt)
    if len(generated_ids) > prompt_len:
        response_ids = generated_ids[prompt_len:]
        response = vocab.decode(response_ids, skip_special=True)
    else:
        response = ""
    
    return response


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """
    Sample the next token ID using temperature and optional top-k filtering.
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_token_id = indices[torch.multinomial(probs, num_samples=1)].item()
    else:
        next_token_id = torch.argmax(logits).item()
    
    return next_token_id


if __name__ == '__main__':
    # Test generation (requires a trained model)
    from vocab import Vocabulary
    from model import FanoosLLMModel
    
    print("Loading vocabulary...")
    vocab = Vocabulary()
    config.VOCAB_SIZE = vocab.size
    
    print("Initializing model...")
    model = FanoosLLMModel(vocab_size=vocab.size)
    
    # Try to load a trained model if it exists
    import os
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading model from {config.MODEL_SAVE_PATH}...")
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded!")
    else:
        print("No trained model found. Please train the model first.")
        print("Using untrained model for testing...")
    
    # Test generation
    test_prompts = [
        "hello",
        "what is your name",
        "what is 3 + 4",
        "compute 5 * 2"
    ]
    
    print("\nTesting generation:")
    print("=" * 60)
    for prompt in test_prompts:
        response = generate_response(prompt, model, vocab)
        print(f"Prompt: '{prompt}'")
        print(f"Response: '{response}'")
        print()

