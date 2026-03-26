"""
Positional Encoding for Transformer Models

Required baseline scope in this file:
1. Sinusoidal positional encoding

Optional extension in this file:
2. Rotary Position Embedding (RoPE)
3. Learned positional embeddings

Positional encodings are crucial for transformers because the attention mechanism
itself is permutation-invariant and doesn't have a notion of token order.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention is All You Need".

    This encoding uses sine and cosine functions of different frequencies to encode
    position information. The encoding is deterministic and doesn't require learning.

    For position pos and dimension i:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - Each dimension corresponds to a sinusoid with wavelength forming a geometric progression
    - The model can easily learn to attend to relative positions
    - Works well for sequences of varying lengths

    Args:
        d_model: Dimension of the model (embedding dimension)
        max_len: Maximum sequence length to pre-compute encodings for
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model) to store positional encodings
        pe = torch.zeros(max_len, d_model)

        # TODO: Create position indices [0, 1, 2, ..., max_len-1]
        # Hint: Use torch.arange() and unsqueeze to shape (max_len, 1)
        position = None  # STUDENT TODO

        # TODO: Create the division term for the sinusoidal functions
        # Formula: div_term = 10000^(2i/d_model) for i in [0, d_model/2)
        # Hint: Use torch.arange(0, d_model, 2) to get even indices
        # Then compute: exp(-log(10000.0) * arange / d_model)
        div_term = None  # STUDENT TODO

        # TODO: Apply sine to even indices (0, 2, 4, ...)
        # Formula: PE(pos, 2i) = sin(pos / div_term)
        # Hint: pe[:, 0::2] selects even columns
        pass  # STUDENT TODO

        # TODO: Apply cosine to odd indices (1, 3, 5, ...)
        # Formula: PE(pos, 2i+1) = cos(pos / div_term)
        # Hint: pe[:, 1::2] selects odd columns
        pass  # STUDENT TODO

        # TODO: Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        pe = None  # STUDENT TODO

        # Register as buffer (not a parameter, but should be saved with the model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added
        """
        # TODO: Add positional encoding to input
        # Hint: self.pe[:, :x.size(1)] selects the encodings for the current sequence length
        # The positional encoding should be added to x
        x = None  # STUDENT TODO

        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    OPTIONAL EXTENSION.

    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    RoPE encodes position information by rotating the query and key representations using
    a rotation matrix. This approach has several advantages:
    - Naturally encodes relative position information
    - Decays with increasing relative distance
    - Can extrapolate to longer sequences than seen during training

    The rotation is applied in the complex plane. For each pair of dimensions (2i, 2i+1),
    we treat them as a complex number and rotate by an angle that depends on the position.

    For position m and dimension pair (2i, 2i+1):
        θ_i = 10000^(-2i/d)
        Rotation angle = m * θ_i

    Args:
        dim: Dimension of each attention head
        max_len: Maximum sequence length
        base: Base for the geometric progression (default: 10000)
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # TODO: Compute inverse frequencies (θ values)
        # Formula: θ_i = base^(-2i/dim) for i in [0, dim/2)
        # Hint: Use torch.arange(0, dim, 2) to get even indices
        # Then compute: base ** (-torch.arange(0, dim, 2).float() / dim)
        inv_freq = None  # STUDENT TODO

        self.register_buffer("inv_freq", inv_freq)

        # Pre-compute rotation matrices for efficiency
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """
        Pre-compute cos and sin values for all positions up to seq_len.

        Args:
            seq_len: Sequence length to cache
        """
        self.max_seq_len_cached = seq_len

        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # Hint: Use torch.arange()
        t = None  # STUDENT TODO

        # TODO: Compute frequencies for each position
        # Formula: freqs = outer_product(t, inv_freq)
        # Hint: Use torch.outer() or einsum('i,j->ij', t, self.inv_freq)
        # Result shape: (seq_len, dim/2)
        freqs = None  # STUDENT TODO

        # TODO: Concatenate frequencies to match the full dimension
        # We need to repeat each frequency for the pair of dimensions it affects
        # Hint: Use torch.cat([freqs, freqs], dim=-1)
        # Result shape: (seq_len, dim)
        emb = None  # STUDENT TODO

        # TODO: Compute cos and sin of the frequencies
        # These will be used to rotate the query and key vectors
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions of the input.

        This is a helper function for applying the rotation. It rearranges the tensor
        so that when we multiply by sin, we get the imaginary part of the rotation.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Rotated tensor of shape (..., dim)
        """
        # TODO: Split x into two halves along the last dimension
        # Hint: Use torch.chunk(x, 2, dim=-1) or x[..., :x.shape[-1]//2] and x[..., x.shape[-1]//2:]
        x1, x2 = None, None  # STUDENT TODO

        # TODO: Concatenate [-x2, x1]
        # This implements the rotation in the complex plane
        # Hint: Use torch.cat([-x2, x1], dim=-1)
        return None  # STUDENT TODO

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Tuple of (rotated_q, rotated_k) with the same shapes as inputs
        """
        seq_len = q.shape[2]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # TODO: Get cos and sin values for the current sequence length
        # Hint: Use self.cos_cached[:seq_len] and self.sin_cached[:seq_len]
        cos = None  # STUDENT TODO
        sin = None  # STUDENT TODO

        # TODO: Reshape cos and sin for broadcasting
        # Current shape: (seq_len, dim)
        # Target shape: (1, 1, seq_len, dim) to broadcast with (batch, heads, seq_len, dim)
        # Hint: Use unsqueeze or view
        cos = None  # STUDENT TODO
        sin = None  # STUDENT TODO

        # TODO: Apply rotation to query
        # Formula: q_rotated = q * cos + rotate_half(q) * sin
        # This implements: q * e^(i*θ) in the complex plane
        q_embed = None  # STUDENT TODO

        # TODO: Apply rotation to key
        # Formula: k_rotated = k * cos + rotate_half(k) * sin
        k_embed = None  # STUDENT TODO

        return q_embed, k_embed


class LearnedPositionalEmbedding(nn.Module):
    """
    OPTIONAL EXTENSION.

    Learned Positional Embedding.

    Instead of using fixed sinusoidal encodings, this approach learns position embeddings
    as parameters during training. This is simpler but requires training data to cover
    the full range of positions.

    Args:
        max_len: Maximum sequence length
        d_model: Dimension of the model
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # TODO: Create an embedding layer for positions
        # Hint: Use nn.Embedding(max_len, d_model)
        self.pos_embedding = None  # STUDENT TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional embeddings added
        """
        batch_size, seq_len, d_model = x.size()

        # TODO: Create position indices [0, 1, 2, ..., seq_len-1]
        # Hint: Use torch.arange(seq_len, device=x.device)
        positions = None  # STUDENT TODO

        # TODO: Get position embeddings and add to input
        # Hint: self.pos_embedding(positions) gives shape (seq_len, d_model)
        # Need to add this to x (broadcasting will handle the batch dimension)
        x = None  # STUDENT TODO

        return x


# Utility function to visualize positional encodings
def visualize_positional_encoding(
    encoding_type: str = "sinusoidal",
    d_model: int = 128,
    max_len: int = 100,
):
    """
    Visualize positional encodings.

    This function is provided for students to understand how different positional
    encodings look. It's not required for the model implementation.

    Args:
        encoding_type: Type of encoding ("sinusoidal" or "learned")
        d_model: Model dimension
        max_len: Maximum sequence length to visualize

    Returns:
        Positional encoding matrix of shape (max_len, d_model)
    """
    import matplotlib.pyplot as plt

    if encoding_type == "sinusoidal":
        pe_module = SinusoidalPositionalEncoding(d_model, max_len)
        # Extract the positional encoding matrix
        pe = pe_module.pe.squeeze(0).numpy()
    else:
        raise ValueError(f"Visualization not implemented for {encoding_type}")

    plt.figure(figsize=(15, 5))
    plt.imshow(pe, cmap="RdBu", aspect="auto")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.colorbar()
    plt.title(f"{encoding_type.capitalize()} Positional Encoding")
    plt.tight_layout()
    plt.savefig(f"positional_encoding_{encoding_type}.png")
    plt.close()

    return pe
