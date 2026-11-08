"""
Neural network model for TinyLLM.
Implements a minimal Transformer Decoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (for causal masking)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply causal mask if provided (for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Feedforward network (2-layer MLP with ReLU activation).
    """
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize feedforward network.
        
        Args:
            embed_dim: Embedding dimension
            ff_dim: Feedforward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feedforward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            Output tensor of same shape
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block (self-attention + feedforward).
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feedforward hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TinyLLMModel(nn.Module):
    """
    Main TinyLLM model: a minimal Transformer Decoder.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = None, num_heads: int = None,
                 num_layers: int = None, ff_dim: int = None, dropout: float = None,
                 max_seq_len: int = None):
        """
        Initialize TinyLLM model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension (uses config if None)
            num_heads: Number of attention heads (uses config if None)
            num_layers: Number of transformer layers (uses config if None)
            ff_dim: Feedforward dimension (uses config if None)
            dropout: Dropout probability (uses config if None)
            max_seq_len: Maximum sequence length (uses config if None)
        """
        super().__init__()
        
        # Use config values if not provided
        embed_dim = embed_dim or config.EMBEDDING_DIM
        num_heads = num_heads or config.NUM_HEADS
        num_layers = num_layers or config.NUM_LAYERS
        ff_dim = ff_dim or config.FEEDFORWARD_DIM
        dropout = dropout or config.DROPOUT
        max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding (learned embeddings)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create a causal mask for decoder (prevents looking at future tokens).
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Mask tensor of shape (1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0)  # (1, seq_len, seq_len)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
        
        Returns:
            Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len)
        # Move mask to same device as input
        mask = mask.to(input_ids.device)
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.positional_encoding(positions)  # (1, seq_len, embed_dim)
        
        # Combine token and positional embeddings
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # (batch_size, seq_len, vocab_size)
        
        return logits


if __name__ == '__main__':
    # Test model
    from vocab import Vocabulary
    
    vocab = Vocabulary()
    model = TinyLLMModel(vocab_size=vocab.size)
    
    # Ensure model is on CPU
    device = torch.device('cpu')
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab.size, (batch_size, seq_len))
    input_ids = input_ids.to(device)
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

