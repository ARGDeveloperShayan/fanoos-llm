"""
Main orchestration script for FanoosLLM.
Trains the model and demonstrates text generation.
"""

import torch
import config
from vocab import Vocabulary
from dataset import FanoosLLMDataset
from model import FanoosLLMModel
from train import train_model
from generate import generate_response


def main():
    """
    Main function: train model and generate examples.
    """
    print("=" * 60)
    print("FANOOS LLM - Minimal From-Scratch Language Model")
    print("=" * 60)
    print()
    
    # Step 1: Initialize vocabulary
    print("Step 1: Initializing vocabulary...")
    vocab = Vocabulary()
    config.VOCAB_SIZE = vocab.size
    vocab.save(config.VOCAB_SAVE_PATH)  # Save vocabulary for web app
    print(f"  Vocabulary size: {vocab.size}")
    print(f"  Sample tokens: {vocab.tokens[:10]}...")
    print(f"  Vocabulary saved to: {config.VOCAB_SAVE_PATH}")
    print()
    
    # Step 2: Create datasets
    print("Step 2: Generating training data...")
    print(f"  Generating {config.TRAIN_DATA_SIZE} training examples...")
    train_dataset = FanoosLLMDataset(vocab, num_examples=config.TRAIN_DATA_SIZE)
    print(f"  Training dataset size: {len(train_dataset)}")
    
    print(f"  Generating {config.VAL_DATA_SIZE} validation examples...")
    val_dataset = FanoosLLMDataset(vocab, num_examples=config.VAL_DATA_SIZE)
    print(f"  Validation dataset size: {len(val_dataset)}")
    print()
    
    # Show some example data
    print("  Sample training examples:")
    for i in range(3):
        input_ids, target_ids = train_dataset[i]
        input_text = vocab.decode(input_ids.tolist())
        target_text = vocab.decode(target_ids.tolist())
        print(f"    {i+1}. Input: '{input_text}' -> Target: '{target_text}'")
    print()
    
    # Step 3: Train model
    print("Step 3: Training model...")
    print(f"  Architecture: Transformer Decoder")
    print(f"  Embedding dim: {config.EMBEDDING_DIM}")
    print(f"  Num layers: {config.NUM_LAYERS}")
    print(f"  Num heads: {config.NUM_HEADS}")
    print(f"  Feedforward dim: {config.FEEDFORWARD_DIM}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Num epochs: {config.NUM_EPOCHS}")
    print()
    
    model = train_model(vocab, train_dataset, val_dataset)
    print()
    
    # Step 4: Load the saved model (to ensure we use the best checkpoint)
    print("Step 4: Loading trained model...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model loaded from {config.MODEL_SAVE_PATH}")
    print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    if 'val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['val_loss']:.4f}")
    print()
    
    # Step 5: Generate example responses
    print("Step 5: Testing text generation...")
    print("=" * 60)
    
    test_prompts = [
        ("hey", "Greeting"),
        ("hello", "Greeting"),
        ("what is your name", "Name query"),
        ("tell me your name please", "Name query with please"),
        ("what is 3 + 4", "Addition"),
        ("what is 9 - 2", "Subtraction"),
        ("what is 5 * 3", "Multiplication"),
        ("what is 8 / 2", "Division"),
        ("compute 4 + 5", "Compute addition"),
        ("solve 6 * 2", "Solve multiplication"),
    ]
    
    print("\nGenerated Responses:")
    print("-" * 60)
    for prompt, description in test_prompts:
        response = generate_response(prompt, model, vocab)
        print(f"Prompt ({description}): '{prompt}'")
        print(f"Response: '{response}'")
        print()
    
    print("=" * 60)
    print("FanoosLLM demonstration completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

