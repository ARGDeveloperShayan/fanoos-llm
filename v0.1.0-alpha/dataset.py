"""
PyTorch Dataset class for TinyLLM.
Handles data loading, tokenization, and formatting.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import config
from vocab import Vocabulary
from data_generator import SyntheticDataGenerator


class TinyLLMDataset(Dataset):
    """
    PyTorch Dataset for TinyLLM training data.
    Generates or loads synthetic data and handles tokenization.
    """
    
    def __init__(self, vocab: Vocabulary, data: List[Tuple[str, str]] = None, 
                 num_examples: int = None, max_seq_len: int = None):
        """
        Initialize the dataset.
        
        Args:
            vocab: Vocabulary instance for tokenization
            data: Pre-generated data (list of (prompt, target) tuples). If None, generates new data.
            num_examples: Number of examples to generate if data is None
            max_seq_len: Maximum sequence length for padding/truncation
        """
        self.vocab = vocab
        self.max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        
        # Generate or use provided data
        if data is None:
            if num_examples is None:
                num_examples = config.TRAIN_DATA_SIZE
            generator = SyntheticDataGenerator(vocab)
            self.data = generator.generate_dataset(num_examples)
        else:
            self.data = data
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.
        
        Args:
            idx: Index of the example
        
        Returns:
            Tuple of (input_ids, target_ids) as torch tensors
        """
        prompt, target = self.data[idx]
        
        # For decoder-only models, we use next-token prediction
        # Input: [BOS] + prompt + target (full sequence)
        # Target: prompt + target + [EOS] (shifted by 1 for next-token prediction)
        
        # Encode prompt and target separately
        prompt_ids = self.vocab.encode(prompt, add_bos=True, add_eos=False)
        target_ids = self.vocab.encode(target, add_bos=False, add_eos=True)
        
        # Combine: [BOS] + prompt + target + [EOS]
        full_sequence = prompt_ids + target_ids
        
        # For next-token prediction:
        # Input: all tokens except the last
        # Target: all tokens except the first (shifted by 1)
        if len(full_sequence) > 1:
            input_ids = full_sequence[:-1]
            target_ids = full_sequence[1:]
        else:
            input_ids = full_sequence
            target_ids = full_sequence
        
        # Pad or truncate both to same length
        input_ids = self._pad_or_truncate(input_ids, self.max_seq_len, pad_value=self.vocab.pad_id)
        target_ids = self._pad_or_truncate(target_ids, self.max_seq_len, pad_value=self.vocab.pad_id)
        
        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return input_tensor, target_tensor
    
    def _pad_or_truncate(self, sequence: List[int], max_len: int, pad_value: int) -> List[int]:
        """
        Pad or truncate a sequence to a fixed length.
        
        Args:
            sequence: List of token IDs
            max_len: Target length
            pad_value: Value to use for padding
        
        Returns:
            Padded/truncated sequence
        """
        if len(sequence) > max_len:
            # Truncate (keep beginning, remove end)
            return sequence[:max_len]
        else:
            # Pad to max_len
            padding = [pad_value] * (max_len - len(sequence))
            return sequence + padding


if __name__ == '__main__':
    # Test dataset
    from vocab import Vocabulary
    
    vocab = Vocabulary()
    dataset = TinyLLMDataset(vocab, num_examples=10)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    input_tensor, target_tensor = dataset[0]
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target shape: {target_tensor.shape}")
    print(f"Input: {vocab.decode(input_tensor.tolist())}")
    print(f"Target: {vocab.decode(target_tensor.tolist())}")

