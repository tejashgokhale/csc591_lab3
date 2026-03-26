"""
Attention Mechanisms for Transformer Models

Required baseline scope in this file:
1. Scaled Dot-Product Attention
2. Multi-Head Attention (MHA)
3. Causal and padding masks

Optional extension in this file:
4. Grouped Query Attention (GQA)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    The attention mechanism computes:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    where:
        Q: Query matrix of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key matrix of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value matrix of shape (batch_size, num_heads, seq_len, head_dim)
        d_k: Dimension of the key vectors (head_dim)

    Args:
        dropout: Dropout probability applied to attention weights
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                  or (batch_size, num_heads, seq_len_q, seq_len_k)
                  In the released baseline path, masks use 1 for valid positions
                  and 0 for masked positions.

        Returns:
            output: Attention output of shape (batch_size, num_heads, seq_len_q, head_dim)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # TODO: Get the dimension of the key vectors (head_dim)
        # Hint: Use query.size(-1) or key.size(-1)
        d_k = None  # STUDENT TODO

        # TODO: Compute attention scores
        # Formula: scores = (Q @ K^T) / sqrt(d_k)
        # Hint: Use torch.matmul() and transpose the last two dimensions of key
        scores = None  # STUDENT TODO

        # TODO: Apply mask if provided
        # Hint: Use scores.masked_fill(mask == 0, float('-inf'))
        # This sets masked positions to -inf so they become 0 after softmax
        if mask is not None:
            pass  # STUDENT TODO

        # TODO: Apply softmax to get attention weights
        # Hint: Apply softmax along the last dimension (dim=-1)
        attention_weights = None  # STUDENT TODO

        # TODO: Apply dropout to attention weights
        attention_weights = None  # STUDENT TODO

        # TODO: Compute the weighted sum of values
        # Formula: output = attention_weights @ V
        output = None  # STUDENT TODO

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Multi-head attention allows the model to jointly attend to information from different
    representation subspaces at different positions.

    The mechanism works as follows:
    1. Project input to Q, K, V using learned linear transformations
    2. Split Q, K, V into multiple heads
    3. Apply scaled dot-product attention for each head in parallel
    4. Concatenate the outputs from all heads
    5. Apply a final linear projection

    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        # TODO: Create linear layers for query, key, and value projections
        # Each should project from d_model to d_model dimensions
        self.q_proj = None  # STUDENT TODO
        self.k_proj = None  # STUDENT TODO
        self.v_proj = None  # STUDENT TODO

        # Output projection
        # TODO: Create a linear layer to project concatenated heads back to d_model
        self.out_proj = None  # STUDENT TODO

        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, d_model = x.size()

        # TODO: Reshape x to (batch_size, seq_len, num_heads, head_dim)
        # Hint: Use x.view()
        x = None  # STUDENT TODO

        # TODO: Transpose to (batch_size, num_heads, seq_len, head_dim)
        # Hint: Use x.transpose(1, 2)
        x = None  # STUDENT TODO

        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine heads back into a single dimension.

        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, head_dim = x.size()

        # TODO: Transpose to (batch_size, seq_len, num_heads, head_dim)
        x = None  # STUDENT TODO

        # TODO: Reshape to (batch_size, seq_len, d_model)
        # Hint: Use x.contiguous().view()
        x = None  # STUDENT TODO

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # TODO: Project query, key, and value
        # Hint: Apply self.q_proj, self.k_proj, self.v_proj
        Q = None  # STUDENT TODO
        K = None  # STUDENT TODO
        V = None  # STUDENT TODO

        # TODO: Split into multiple heads
        # Hint: Use self._split_heads()
        Q = None  # STUDENT TODO
        K = None  # STUDENT TODO
        V = None  # STUDENT TODO

        # TODO: Apply attention
        # Hint: Use self.attention()
        attn_output, attention_weights = None, None  # STUDENT TODO

        # TODO: Combine heads
        # Hint: Use self._combine_heads()
        attn_output = None  # STUDENT TODO

        # TODO: Apply output projection
        # Hint: Use self.out_proj()
        output = None  # STUDENT TODO

        return output, attention_weights


class GroupedQueryAttention(nn.Module):
    """
    OPTIONAL EXTENSION.

    Grouped Query Attention (GQA) mechanism.

    GQA is a more memory-efficient variant of multi-head attention that reduces the number
    of key-value heads while maintaining the same number of query heads. This reduces the
    KV cache size during inference, making it more efficient for deployment.

    In standard MHA: num_kv_heads = num_heads
    In GQA: num_kv_heads < num_heads (typically num_heads // 2 or num_heads // 4)
    In MQA (Multi-Query Attention): num_kv_heads = 1

    Each KV head is shared across multiple query heads.

    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (must divide num_heads evenly)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Linear projections
        # TODO: Create query projection (d_model -> d_model)
        self.q_proj = None  # STUDENT TODO

        # TODO: Create key projection (d_model -> num_kv_heads * head_dim)
        # Note: KV heads have reduced dimension compared to query heads
        self.k_proj = None  # STUDENT TODO

        # TODO: Create value projection (d_model -> num_kv_heads * head_dim)
        self.v_proj = None  # STUDENT TODO

        # TODO: Create output projection
        self.out_proj = None  # STUDENT TODO

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, head_dim) and transpose.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_heads * head_dim)
            num_heads: Number of heads to split into

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat key or value heads to match the number of query heads.

        This is the key operation that makes GQA work: each KV head is repeated
        to be used by multiple query heads.

        Args:
            x: Input tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)

        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_kv_heads, seq_len, head_dim = x.size()

        if self.num_queries_per_kv == 1:
            return x

        # TODO: Repeat each KV head num_queries_per_kv times
        # Hint: Use x.unsqueeze(2) to add a dimension, then repeat_interleave
        # Shape progression:
        # (batch, num_kv_heads, seq_len, head_dim)
        # -> (batch, num_kv_heads, 1, seq_len, head_dim)
        # -> (batch, num_kv_heads, num_queries_per_kv, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        x = None  # STUDENT TODO

        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine heads back into a single dimension."""
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, self.d_model)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of grouped query attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor

        Returns:
            output: Attention output of shape (batch_size, seq_len_q, d_model)
            attention_weights: Attention weights
        """
        # TODO: Project query, key, and value
        Q = None  # STUDENT TODO
        K = None  # STUDENT TODO
        V = None  # STUDENT TODO

        # TODO: Split into heads
        # Note: Q has num_heads, but K and V have num_kv_heads
        Q = None  # STUDENT TODO (use self.num_heads)
        K = None  # STUDENT TODO (use self.num_kv_heads)
        V = None  # STUDENT TODO (use self.num_kv_heads)

        # TODO: Repeat K and V to match the number of query heads
        # Hint: Use self._repeat_kv()
        K = None  # STUDENT TODO
        V = None  # STUDENT TODO

        # TODO: Apply attention (same as MHA after repeating KV)
        attn_output, attention_weights = None, None  # STUDENT TODO

        # TODO: Combine heads and apply output projection
        attn_output = None  # STUDENT TODO
        output = None  # STUDENT TODO

        return output, attention_weights


# Helper function to create causal mask
def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (lower triangular) mask for autoregressive attention.

    The mask ensures that position i can only attend to positions <= i.

    Args:
        seq_len: Sequence length
        device: Device to create the mask on

    Returns:
        Mask tensor of shape (1, 1, seq_len, seq_len) with 1s in lower triangle and 0s above
    """
    # TODO: Create a lower triangular matrix of ones
    # Hint: Use torch.tril(torch.ones(...))
    mask = None  # STUDENT TODO

    # TODO: Reshape to (1, 1, seq_len, seq_len) for broadcasting
    mask = None  # STUDENT TODO

    return mask


# Helper function to create padding mask
def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create a padding mask from a sequence with padding tokens.

    Args:
        seq: Input sequence of shape (batch_size, seq_len) with token indices
        pad_idx: Index of the padding token

    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len) with 1s for real tokens and 0s for padding
    """
    # TODO: Create mask where padding tokens are 0 and real tokens are 1
    # Hint: Use (seq != pad_idx)
    mask = None  # STUDENT TODO

    # TODO: Reshape to (batch_size, 1, 1, seq_len) for broadcasting
    mask = None  # STUDENT TODO

    return mask
