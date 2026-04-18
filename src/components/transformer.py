"""
Transformer Encoder and Decoder Layers

Required baseline scope in this file:
1. Transformer Decoder Layer with causal masking
2. Residual connections
3. One working norm/FFN/attention composition path

Optional extension in this file:
4. Transformer Encoder Layer
5. cross-attention support
6. extra architecture variants such as GQA-backed decoder layers
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, create_causal_mask
from .feedforward import create_ffn
from .normalization import LayerNorm, RMSNorm


class TransformerEncoderLayer(nn.Module):
    """
    OPTIONAL EXTENSION.

    Transformer Encoder Layer.

    The encoder layer consists of:
    1. Multi-head self-attention
    2. Feed-forward network
    3. Residual connections around each sub-layer
    4. Layer normalization (pre-norm or post-norm)

    Pre-norm architecture (modern, more stable):
        x = x + attention(norm(x))
        x = x + ffn(norm(x))

    Post-norm architecture (original Transformer):
        x = norm(x + attention(x))
        x = norm(x + ffn(x))

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
        activation: Activation function for FFN
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        norm_position: Position of normalization ("pre" or "post")
        attention_type: Type of attention ("mha" or "gqa")
        num_kv_heads: Number of key-value heads (for GQA)
        ffn_type: Type of feed-forward network
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        attention_type: str = "mha",
        num_kv_heads: Optional[int] = None,
        ffn_type: str = "standard",
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_position = norm_position

        # TODO: Create attention mechanism
        # Hint: Use MultiHeadAttention if attention_type == "mha"
        #       Use GroupedQueryAttention if attention_type == "gqa"
        if attention_type == "mha":
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)  # STUDENT TODO
        elif attention_type == "gqa":
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2  # Default to half
            self.attention = None  # STUDENT TODO
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        # TODO: Create feed-forward network
        # Hint: Use create_ffn() function
        self.ffn = create_ffn(ffn_type, d_model, d_ff, dropout, activation)  # STUDENT TODO

        # TODO: Create normalization layers
        # We need two normalization layers: one for attention, one for FFN
        # Hint: Use LayerNorm(d_model) if norm_type == "layernorm"
        #       Use RMSNorm(d_model) if norm_type == "rmsnorm"
        if norm_type == "layernorm":
            self.norm1 = LayerNorm(d_model)  # STUDENT TODO
            self.norm2 = LayerNorm(d_model)  # STUDENT TODO
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)  # STUDENT TODO
            self.norm2 = RMSNorm(d_model)  # STUDENT TODO
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self_attn_mask is None:
            self_attn_mask = create_causal_mask(x.size(1), x.device)

        if self.norm_position == "pre":
            # Pre-norm: norm -> attention -> residual
            # TODO: Apply self-attention with pre-norm
            # 1. Normalize input: norm1(x)
            # 2. Apply attention: self.attention(normalized, normalized, normalized, mask)
            # 3. Apply dropout to attention output
            # 4. Add residual connection: x + dropout(attention_output)
            normed = self.norm1(x)  # STUDENT TODO (normalize)
            attn_output, _ = self.self_attention(normed, normed, normed, self_attn_mask)  # STUDENT TODO (attention)
            x = x + self.dropout(attn_output)  # STUDENT TODO (residual: x + dropout(attn_output))

            # Pre-norm: norm -> ffn -> residual
            # TODO: Apply feed-forward with pre-norm
            # 1. Normalize: norm2(x)
            # 2. Apply FFN: self.ffn(normalized)
            # 3. Apply dropout
            # 4. Add residual connection: x + dropout(ffn_output)
            normed = self.norm2(x)  # STUDENT TODO
            ffn_output = self.ffn(normed)  # STUDENT TODO
            x = x + self.dropout(ffn_output)  # STUDENT TODO (residual)

        else:  # post-norm
            # Post-norm: attention -> residual -> norm
            # TODO: Apply self-attention with post-norm
            # 1. Apply attention: self.attention(x, x, x, mask)
            # 2. Apply dropout
            # 3. Add residual: x + dropout(attention_output)
            # 4. Normalize: norm1(x)
            attn_output, _ = None, None  # STUDENT TODO
            x = None  # STUDENT TODO (residual)
            x = None  # STUDENT TODO (normalize)

            # Post-norm: ffn -> residual -> norm
            # TODO: Apply feed-forward with post-norm
            # 1. Apply FFN: self.ffn(x)
            # 2. Apply dropout
            # 3. Add residual: x + dropout(ffn_output)
            # 4. Normalize: norm2(x)
            ffn_output = None  # STUDENT TODO
            x = None  # STUDENT TODO (residual)
            x = None  # STUDENT TODO (normalize)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.

    The decoder layer consists of:
    1. Masked multi-head self-attention (causal)
    2. Optional cross-attention (for encoder-decoder models)
    3. Feed-forward network
    4. Residual connections around each sub-layer
    5. Layer normalization

    For autoregressive language models (decoder-only), we only use masked self-attention.

    Causal masking ensures that position i can only attend to positions <= i,
    which is necessary for autoregressive generation.

    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability
        activation: Activation function for FFN
        norm_type: Type of normalization
        norm_position: Position of normalization ("pre" or "post")
        attention_type: Type of attention ("mha" or "gqa")
        num_kv_heads: Number of key-value heads (for GQA)
        ffn_type: Type of feed-forward network
        use_cross_attention: Whether to include cross-attention (for encoder-decoder)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_position: str = "pre",
        attention_type: str = "mha",
        num_kv_heads: Optional[int] = None,
        ffn_type: str = "standard",
        use_cross_attention: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm_position = norm_position
        self.use_cross_attention = use_cross_attention

        # TODO: Create self-attention mechanism (same as encoder)
        if attention_type == "mha":
            self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)  # STUDENT TODO
        elif attention_type == "gqa":
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2
            self.self_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)  # STUDENT TODO
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        # TODO: Create cross-attention if needed (for encoder-decoder models)
        if use_cross_attention:
            if attention_type == "mha":
                self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)  # STUDENT TODO
            elif attention_type == "gqa":
                self.cross_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)  # STUDENT TODO

        # TODO: Create feed-forward network
        self.ffn = create_ffn(ffn_type, d_model, d_ff, dropout, activation)  # STUDENT TODO

        # TODO: Create normalization layers
        # We need 2 or 3 norm layers depending on whether we use cross-attention
        if norm_type == "layernorm":
            self.norm1 = LayerNorm(d_model)  # STUDENT TODO (for self-attention)
            if use_cross_attention:
                self.norm2 = LayerNorm(d_model)  # STUDENT TODO (for cross-attention)
            self.norm3 = LayerNorm(d_model)  # STUDENT TODO (for FFN)
        elif norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)  # STUDENT TODO
            if use_cross_attention:
                self.norm2 = RMSNorm(d_model)  # STUDENT TODO
            self.norm3 = RMSNorm(d_model)  # STUDENT TODO
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of decoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Optional encoder output for cross-attention
                           Shape: (batch_size, src_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Optional mask for cross-attention

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Create causal mask if not provided
        # Hint: Use create_causal_mask(x.size(1), x.device)
        if self_attn_mask is None:
            self_attn_mask = create_causal_mask(x.size(1), x.device)  # STUDENT TODO

        if self.norm_position == "pre":
            # Pre-norm: Masked self-attention
            # TODO: Apply masked self-attention with pre-norm
            # Same as encoder, but use self_attn_mask (causal mask)
            normed = self.norm1(x)  # STUDENT TODO
            attn_output, _ = self.self_attention(normed, normed, normed, self_attn_mask)  # STUDENT TODO
            x = x + self.dropout(attn_output)  # STUDENT TODO (residual)

            # Pre-norm: Cross-attention (if enabled)
            if self.use_cross_attention and encoder_output is not None:
                # TODO: Apply cross-attention with pre-norm
                # Query comes from decoder (x), Key and Value come from encoder
                # Hint: self.cross_attention(query=normed, key=encoder_output, value=encoder_output)
                normed = self.norm2(x)  # STUDENT TODO
                cross_attn_output, _ = self.cross_attention(
                normed, encoder_output, encoder_output, cross_attn_mask
            )  # STUDENT TODO
                x = x + self.dropout(cross_attn_output)  # STUDENT TODO (residual)

            # Pre-norm: Feed-forward
            # TODO: Apply FFN with pre-norm
            normed = self.norm3(x)  # STUDENT TODO
            ffn_output = self.ffn(normed)  # STUDENT TODO
            x = x + self.dropout(ffn_output)  # STUDENT TODO (residual)

        else:  # post-norm
            # Post-norm: Masked self-attention
            # TODO: Apply masked self-attention with post-norm
            attn_output, _ = self.self_attention(x, x, x, self_attn_mask)  # STUDENT TODO
            x = x + self.dropout(attn_output)  # STUDENT TODO (residual)
            x = self.norm1(x)  # STUDENT TODO (normalize)

            # Post-norm: Cross-attention (if enabled)
            if self.use_cross_attention and encoder_output is not None:
                # TODO: Apply cross-attention with post-norm
                cross_attn_output, _ = self.cross_attention(
                x, encoder_output, encoder_output, cross_attn_mask
            )  # STUDENT TODO
                x = x + self.dropout(cross_attn_output)  # STUDENT TODO (residual)
                x = self.norm2(x)  # STUDENT TODO (normalize)

            # Post-norm: Feed-forward
            # TODO: Apply FFN with post-norm
            ffn_output = self.ffn(x)  # STUDENT TODO
            x = x + self.dropout(ffn_output)  # STUDENT TODO (residual)
            x = self.norm3(x)  # STUDENT TODO (normalize)

        return x


# Utility function to visualize attention masks
def visualize_causal_mask(seq_len: int = 10, save_path: str = "causal_mask.png"):
    """
    Visualize the causal attention mask.

    This helps students understand how causal masking works in autoregressive models.

    Args:
        seq_len: Sequence length
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt

    # Create causal mask
    mask = create_causal_mask(seq_len, torch.device("cpu"))
    mask = mask.squeeze().numpy()

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="RdYlGn", vmin=0, vmax=1)
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Causal Attention Mask\n(Green = Can Attend, Red = Masked)")
    plt.colorbar(label="Attention Allowed")

    # Add grid
    for i in range(seq_len + 1):
        plt.axhline(i - 0.5, color="black", linewidth=0.5)
        plt.axvline(i - 0.5, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Causal mask visualization saved to {save_path}")
    print("\nInterpretation:")
    print("- Each row represents a query position")
    print("- Each column represents a key position")
    print("- Green cells: query can attend to key (mask = 1)")
    print("- Red cells: query cannot attend to key (mask = 0)")
    print("- Lower triangular pattern ensures causality: position i can only see positions <= i")


# Test function
def test_transformer_layers():
    """
    Test transformer encoder and decoder layers.

    This function helps students verify their implementations.
    """
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads, d_ff = 8, 256

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Transformer Layers:")
    print(f"Input shape: {x.shape}\n")

    # Test encoder layer
    print("1. Encoder Layer (Pre-norm, LayerNorm, MHA):")
    encoder = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="layernorm",
        attention_type="mha",
    )
    out_encoder = encoder(x)
    print(f"   Output shape: {out_encoder.shape}")

    # Test encoder with GQA
    print("\n2. Encoder Layer (Pre-norm, RMSNorm, GQA):")
    encoder_gqa = TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="rmsnorm",
        attention_type="gqa",
        num_kv_heads=4,
    )
    out_encoder_gqa = encoder_gqa(x)
    print(f"   Output shape: {out_encoder_gqa.shape}")

    # Test decoder layer
    print("\n3. Decoder Layer (Pre-norm, LayerNorm, MHA, Causal):")
    decoder = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        norm_position="pre",
        norm_type="layernorm",
        attention_type="mha",
    )
    out_decoder = decoder(x)
    print(f"   Output shape: {out_decoder.shape}")

    # Test decoder with cross-attention
    print("\n4. Decoder Layer (with Cross-Attention):")
    decoder_cross = TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        use_cross_attention=True,
    )
    encoder_out = torch.randn(batch_size, 15, d_model)  # Different seq_len
    out_decoder_cross = decoder_cross(x, encoder_output=encoder_out)
    print(f"   Output shape: {out_decoder_cross.shape}")

    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    print(f"\nParameter counts:")
    print(f"Encoder (MHA): {count_params(encoder):,}")
    print(f"Encoder (GQA): {count_params(encoder_gqa):,}")
    print(f"Decoder (no cross-attn): {count_params(decoder):,}")
    print(f"Decoder (with cross-attn): {count_params(decoder_cross):,}")
