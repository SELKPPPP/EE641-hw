"""
Positional encoding implementations for length extrapolation analysis.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This encoding can extrapolate to sequence lengths beyond training.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length for precomputation
        """
        super().__init__()
        self.d_model = d_model

        # TODO: Create positional encoding matrix
        # Shape should be [max_len, d_model]
        # Use the sinusoidal formula for positions
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape [1, max_len, d_model]

        # TODO: Register as buffer (not trainable parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)

        # TODO: Add positional encoding to input
        # For sequences longer than max_len, compute sinusoidal values on-the-fly
        # Use the same formula from __init__ to compute positions dynamically

        if seq_len > self.pe.size(1):
            position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # Shape [1, seq_len, d_model]
        else:
            pe = self.pe[:, :seq_len, :]

        return x + pe


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings.

    Each position gets a learnable embedding vector.
    Cannot extrapolate beyond max_len seen during training.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize learned positional embeddings.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # TODO: Create learnable position embeddings
        # Use nn.Embedding with max_len positions
        self.position_embeddings = nn.Embedding(max_len, d_model)


        # TODO: Initialize embeddings (e.g., normal distribution)
        nn.init.normal_(self.position_embeddings.weight, mean=0, std=0.1)
    

    def forward(self, x):
        """
        Add learned positional embeddings to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional embeddings
        """
        batch_size, seq_len, _ = x.size()

        # TODO: Get position indices using torch.arange(seq_len, device=x.device)
        # For extrapolation: use torch.clamp(positions, max=self.max_len-1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = torch.clamp(positions, max=self.max_len - 1)

        # TODO: Look up position embeddings using self.position_embeddings
        position_embeddings = self.position_embeddings(positions)
        position_embeddings = position_embeddings.expand(batch_size, -1, -1)


        # TODO: Add to input and return
        return x + position_embeddings


class NoPositionalEncoding(nn.Module):
    """
    Baseline: No positional encoding.

    Model is permutation-invariant without position information.
    Should fail on position-dependent tasks like sorting detection.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize no-op positional encoding.

        Args:
            d_model: Embedding dimension (unused)
            max_len: Maximum sequence length (unused)
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Return input unchanged.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x unchanged
        """
        # TODO: Return input without modification
        return x


def get_positional_encoding(encoding_type, d_model, max_len=5000):
    """
    Factory function for positional encoding modules.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        d_model: Model dimension
        max_len: Maximum sequence length

    Returns:
        Positional encoding module
    """
    encodings = {
        'sinusoidal': SinusoidalPositionalEncoding,
        'learned': LearnedPositionalEncoding,
        'none': NoPositionalEncoding
    }

    if encoding_type not in encodings:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

    return encodings[encoding_type](d_model, max_len)


def visualize_positional_encoding(encoding_module, max_len=128, d_model=128):
    """
    Visualize positional encoding patterns.

    Args:
        encoding_module: Positional encoding module
        max_len: Number of positions to visualize
        d_model: Model dimension

    Returns:
        Encoding matrix [max_len, d_model] for visualization
    """
    # Create dummy input
    dummy_input = torch.zeros(1, max_len, d_model)

    # Get encoding
    with torch.no_grad():
        encoded = encoding_module(dummy_input)
        encoding = encoded - dummy_input  # Extract just the positional component

    return encoding.squeeze(0).numpy()