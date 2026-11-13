"""
Training script for FanoosLLM.
Implements the training loop with loss calculation, optimization, and validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from vocab import Vocabulary
from dataset import FanoosLLMDataset
from model import FanoosLLMModel


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device,
                scaler: torch.cuda.amp.GradScaler = None) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (CPU)
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        # Move to device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_ids.view(-1)
            
            loss = criterion(logits_flat, target_flat)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, scaler: torch.cuda.amp.GradScaler = None) -> float:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (CPU)
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            # Move to device
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(input_ids)
                
                # Reshape for loss
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target_ids.view(-1)
                
                loss = criterion(logits_flat, target_flat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(vocab: Vocabulary, train_dataset: FanoosLLMDataset,
                val_dataset: FanoosLLMDataset = None) -> FanoosLLMModel:
    """
    Main training function.
    
    Args:
        vocab: Vocabulary instance
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
    
    Returns:
        Trained model
    """
    # Pick device with CUDA preference
    preferred = config.DEVICE
    device = torch.device(preferred if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Training on device: {device}")
    
    # Initialize model
    model = FanoosLLMModel(vocab_size=vocab.size)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        num_workers=0
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
            num_workers=0
        )
    
    # Loss function (Cross-Entropy)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    
    # AMP scaler (only if CUDA and enabled)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and getattr(config, "USE_AMP", False)) else None
    
    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device, scaler)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss! Saving model...")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab.size,
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                }, config.MODEL_SAVE_PATH)
        else:
            # Save model every epoch if no validation
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab.size,
                'epoch': epoch + 1,
            }, config.MODEL_SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    
    return model


if __name__ == '__main__':
    # Test training setup
    from vocab import Vocabulary
    from dataset import FanoosLLMDataset
    
    print("Initializing vocabulary...")
    vocab = Vocabulary()
    config.VOCAB_SIZE = vocab.size
    
    print(f"Vocabulary size: {vocab.size}")
    
    print("Creating datasets...")
    train_dataset = FanoosLLMDataset(vocab, num_examples=100)
    val_dataset = FanoosLLMDataset(vocab, num_examples=20)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    print("Starting training...")
    model = train_model(vocab, train_dataset, val_dataset)
    
    print("Training test completed!")