"""
Normalization Layers for Transformer Models

Required baseline scope in this file:
1. Layer Normalization (LayerNorm)
2. One working residual + normalization path

Optional extension in this file:
3. Root Mean Square Layer Normalization (RMSNorm)

Normalization is crucial for training stability and convergence speed in deep networks.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    LayerNorm normalizes the inputs across the feature dimension for each sample
    independently. It computes the mean and variance across all features for each
    sample in the batch.

    Formula:
        y = (x - mean) / sqrt(var + eps) * gamma + beta

    where:
        mean = mean(x) across feature dimension
        var = variance(x) across feature dimension
        gamma, beta = learnable parameters (scale and shift)
        eps = small constant for numerical stability

    Properties:
    - Normalizes each sample independently (unlike BatchNorm)
    - Works well with variable sequence lengths
    - Helps with gradient flow in deep networks

    Args:
        normalized_shape: Shape of the input features to normalize (typically d_model)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn gamma and beta parameters
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # TODO: Create learnable scale parameter (gamma) initialized to ones
            # Hint: Use nn.Parameter(torch.ones(normalized_shape))
            self.weight = nn.Parameter(torch.ones(normalized_shape))  # STUDENT TODO

            # TODO: Create learnable shift parameter (beta) initialized to zeros
            # Hint: Use nn.Parameter(torch.zeros(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # STUDENT TODO
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model)

        Returns:
            Normalized tensor of the same shape as input
        """
        # TODO: Compute mean across the last dimension (feature dimension)
        # Hint: Use torch.mean(x, dim=-1, keepdim=True)
        # keepdim=True preserves the dimension for broadcasting
        mean = torch.mean(x, dim=-1, keepdim=True)  # STUDENT TODO

        # TODO: Compute variance across the last dimension
        # Hint: Use torch.var(x, dim=-1, keepdim=True, unbiased=False)
        # unbiased=False uses N instead of N-1 in the denominator
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)  # STUDENT TODO

        # TODO: Normalize: (x - mean) / sqrt(var + eps)
        # Hint: Use torch.sqrt() or var.sqrt(), and add self.eps for stability
        x_norm = (x - mean) / torch.sqrt(var + self.eps)  # STUDENT TODO

        # TODO: Apply affine transformation if enabled
        # Formula: x_norm * gamma + beta
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias  # STUDENT TODO

        return x_norm

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"


class RMSNorm(nn.Module):
    """
    OPTIONAL EXTENSION.

    Root Mean Square Layer Normalization.

    RMSNorm is a simpler and more efficient variant of LayerNorm that only uses
    the root mean square (RMS) for normalization, without centering (no mean subtraction).

    Formula:
        y = x / RMS(x) * gamma

    where:
        RMS(x) = sqrt(mean(x^2) + eps)
        gamma = learnable scale parameter
        eps = small constant for numerical stability

    Advantages over LayerNorm:
    - Simpler computation (no mean calculation and subtraction)
    - Fewer parameters (no bias term)
    - Faster training and inference
    - Similar or better performance in practice

    Used in: LLaMA, GPT-NeoX, and other modern LLMs

    Args:
        normalized_shape: Shape of the input features to normalize (typically d_model)
        eps: Small constant for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

        # TODO: Create learnable scale parameter (gamma) initialized to ones
        # Hint: Use nn.Parameter(torch.ones(normalized_shape))
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # STUDENT TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model)

        Returns:
            Normalized tensor of the same shape as input
        """
        # TODO: Compute RMS (Root Mean Square)
        # Formula: RMS = sqrt(mean(x^2) + eps)
        # Step 1: Square the input
        # Step 2: Compute mean across the last dimension (keepdim=True)
        # Step 3: Add eps and take square root
        # Hint: Use torch.mean() and torch.sqrt() or .rsqrt() for 1/sqrt
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)  # STUDENT TODO

        # TODO: Normalize by RMS
        # Formula: x / RMS
        x_norm = x / rms  # STUDENT TODO

        # TODO: Apply scale parameter
        # Formula: x_norm * gamma
        x_norm = x_norm * self.weight  # STUDENT TODO

        return x_norm

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"{self.normalized_shape}, eps={self.eps}"


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper.

    In pre-norm architecture, normalization is applied BEFORE the sub-layer (attention or FFN).
    This is the architecture used in modern transformers (GPT, LLaMA, etc.).

    Structure:
        x = x + sublayer(norm(x))

    Advantages:
    - More stable training
    - Can train deeper models
    - Better gradient flow

    Args:
        dim: Model dimension
        fn: The sub-layer function (attention or FFN)
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
    """

    def __init__(self, dim: int, fn: nn.Module, norm_type: str = "layernorm"):
        super().__init__()
        # TODO: Create normalization layer based on norm_type
        # Hint: Use LayerNorm(dim) if norm_type == "layernorm", else RMSNorm(dim)
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim)  # STUDENT TODO
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)  # STUDENT TODO
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply pre-normalization.

        Args:
            x: Input tensor
            **kwargs: Additional arguments for the sub-layer

        Returns:
            Output tensor after applying norm -> sub-layer -> residual
        """
        # TODO: Apply normalization before the sub-layer
        # Formula: x + fn(norm(x))
        # Hint: self.norm(x) normalizes, self.fn() applies the sub-layer
        return None  # STUDENT TODO


class PostNorm(nn.Module):
    """
    Post-normalization wrapper.

    In post-norm architecture, normalization is applied AFTER the sub-layer (attention or FFN).
    This is the architecture used in the original Transformer paper.

    Structure:
        x = norm(x + sublayer(x))

    Disadvantages compared to pre-norm:
    - Less stable training for deep models
    - May require careful learning rate tuning
    - Gradient flow can be problematic

    Args:
        dim: Model dimension
        fn: The sub-layer function (attention or FFN)
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
    """

    def __init__(self, dim: int, fn: nn.Module, norm_type: str = "layernorm"):
        super().__init__()
        # TODO: Create normalization layer based on norm_type
        if norm_type == "layernorm":
            self.norm = LayerNorm(dim)  # STUDENT TODO
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(dim)  # STUDENT TODO
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply post-normalization.

        Args:
            x: Input tensor
            **kwargs: Additional arguments for the sub-layer

        Returns:
            Output tensor after applying sub-layer -> residual -> norm
        """
        # TODO: Apply normalization after the sub-layer
        # Formula: norm(x + fn(x))
        # Hint: self.fn(x) applies the sub-layer, then add x, then normalize
        out = self.fn(x, **kwargs)
        if isinstance(out, tuple):
            return (self.norm(x + out[0]),) + out[1:]
        return self.norm(x + out)  # STUDENT TODO


# Comparison utility
def compare_normalizations(
    batch_size: int = 2,
    seq_len: int = 10,
    d_model: int = 64,
):
    """
    Compare LayerNorm and RMSNorm behavior.

    This function is provided for students to understand the differences between
    LayerNorm and RMSNorm. It's not required for the model implementation.

    Args:
        batch_size: Batch size for test input
        seq_len: Sequence length for test input
        d_model: Model dimension

    Returns:
        Dictionary with statistics about both normalization methods
    """
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)

    # Apply LayerNorm
    ln = LayerNorm(d_model)
    x_ln = ln(x)

    # Apply RMSNorm
    rms = RMSNorm(d_model)
    x_rms = rms(x)

    # Compute statistics
    stats = {
        "input_mean": x.mean().item(),
        "input_std": x.std().item(),
        "layernorm_mean": x_ln.mean().item(),
        "layernorm_std": x_ln.std().item(),
        "rmsnorm_mean": x_rms.mean().item(),
        "rmsnorm_std": x_rms.std().item(),
    }

    print("Normalization Comparison:")
    print(f"Input - Mean: {stats['input_mean']:.4f}, Std: {stats['input_std']:.4f}")
    print(f"LayerNorm - Mean: {stats['layernorm_mean']:.4f}, Std: {stats['layernorm_std']:.4f}")
    print(f"RMSNorm - Mean: {stats['rmsnorm_mean']:.4f}, Std: {stats['rmsnorm_std']:.4f}")
    print("\nNote: LayerNorm centers the output (mean ≈ 0), while RMSNorm does not.")

    return stats
