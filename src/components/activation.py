"""
Activation Functions for Transformer Models

This module implements various activation functions used in transformer architectures:
1. ReLU (Rectified Linear Unit)
2. GELU (Gaussian Error Linear Unit)
3. SiLU/Swish (Sigmoid Linear Unit)

Students will implement these functions and understand their properties and trade-offs.
"""

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.

    Formula:
        ReLU(x) = max(0, x)

    Properties:
    - Simple and computationally efficient
    - Non-linear but piecewise linear
    - Can suffer from "dying ReLU" problem (neurons that always output 0)
    - Gradient is 1 for x > 0, 0 for x <= 0

    Used in: Original Transformer, many CNNs
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of the same shape
        """
        # TODO: Implement ReLU
        # Hint: Use torch.relu(x) or F.relu(x) or torch.maximum(x, torch.zeros_like(x))
        return None  # STUDENT TODO


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth approximation to ReLU that weights inputs by their magnitude
    rather than gating them by their sign.

    Exact formula:
        GELU(x) = x * Φ(x)
    where Φ(x) is the cumulative distribution function of the standard normal distribution.

    Approximation (used in practice):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Properties:
    - Smooth and differentiable everywhere
    - Non-monotonic (has a small negative region)
    - Better gradient flow than ReLU
    - Empirically performs better in many transformer models

    Used in: BERT, GPT-2, GPT-3, many modern transformers
    """

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of the same shape
        """
        # TODO: Implement GELU
        # Option 1: Use PyTorch's built-in F.gelu(x, approximate=self.approximate)
        # Option 2: Implement the approximation formula:
        #   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # Hint: sqrt(2/π) ≈ 0.7978845608
        return None  # STUDENT TODO


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (SiLU), also known as Swish activation function.

    Formula:
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Properties:
    - Smooth and non-monotonic
    - Self-gated (uses its own value for gating)
    - Unbounded above, bounded below
    - Better gradient flow than ReLU
    - Similar performance to GELU in practice

    Used in: LLaMA, many modern LLMs, EfficientNet

    Note: SiLU and Swish are the same function. PyTorch calls it SiLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SiLU/Swish activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor of the same shape
        """
        # TODO: Implement SiLU
        # Formula: x * sigmoid(x)
        # Hint: Use torch.sigmoid(x) or F.silu(x)
        return None  # STUDENT TODO


class GLU(nn.Module):
    """
    Gated Linear Unit (GLU) activation function.

    GLU splits the input into two halves and uses one half to gate the other.

    Formula:
        GLU(x) = (x * W + b) ⊗ σ(x * V + c)
    where ⊗ is element-wise multiplication and σ is sigmoid.

    In practice, the input is already split, so:
        GLU(a, b) = a ⊗ σ(b)

    Properties:
    - Provides a gating mechanism
    - Can control information flow
    - Used as part of feed-forward networks

    This is typically used in combination with linear layers in the FFN.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU activation.

        Args:
            x: Input tensor of shape (..., 2*dim)
                The input is expected to be split into two halves along the last dimension

        Returns:
            Output tensor of shape (..., dim)
        """
        # TODO: Split input into two halves along the last dimension
        # Hint: Use torch.chunk(x, 2, dim=-1)
        a, b = None, None  # STUDENT TODO

        # TODO: Apply gating: a * sigmoid(b)
        # Hint: Use torch.sigmoid(b)
        return None  # STUDENT TODO


# Activation function factory
def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Name of the activation function ("relu", "gelu", "silu", "glu")

    Returns:
        Activation function module

    Raises:
        ValueError: If activation name is not recognized
    """
    # TODO: Return the appropriate activation function based on name
    # Hint: Use a dictionary or if-elif-else
    activations = {
        "relu": None,  # STUDENT TODO
        "gelu": None,  # STUDENT TODO
        "silu": None,  # STUDENT TODO
        "swish": None,  # STUDENT TODO (same as silu)
        "glu": None,  # STUDENT TODO
    }

    if name.lower() not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Choose from {list(activations.keys())}"
        )

    return activations[name.lower()]


# Visualization utilities
def visualize_activations(
    x_range: tuple = (-5, 5),
    num_points: int = 1000,
    save_path: str = "activation_functions.png",
):
    """
    Visualize different activation functions.

    This function helps students understand the behavior of different activations
    by plotting them over a range of input values.

    Args:
        x_range: Range of x values to plot (min, max)
        num_points: Number of points to sample
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create input values
    x = torch.linspace(x_range[0], x_range[1], num_points)

    # Compute activations
    activations = {
        "ReLU": ReLU()(x),
        "GELU": GELU()(x),
        "SiLU": SiLU()(x),
    }

    # Plot
    plt.figure(figsize=(12, 4))

    # Plot activation functions
    plt.subplot(1, 2, 1)
    for name, y in activations.items():
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.title("Activation Functions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    # Plot derivatives
    plt.subplot(1, 2, 2)
    x_grad = x.clone().requires_grad_(True)
    for name in activations.keys():
        if name == "ReLU":
            y = ReLU()(x_grad)
        elif name == "GELU":
            y = GELU()(x_grad)
        elif name == "SiLU":
            y = SiLU()(x_grad)

        # Compute gradient
        y.sum().backward()
        grad = x_grad.grad.clone()
        x_grad.grad.zero_()

        plt.plot(x.numpy(), grad.numpy(), label=f"{name}'", linewidth=2)

    plt.xlabel("Input (x)")
    plt.ylabel("Gradient")
    plt.title("Activation Function Derivatives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.axvline(x=0, color="k", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Activation functions visualization saved to {save_path}")


def compare_activation_properties():
    """
    Compare properties of different activation functions.

    This function provides a summary of key properties for students to understand
    the trade-offs between different activations.
    """
    properties = {
        "ReLU": {
            "Smooth": False,
            "Bounded Below": True,
            "Bounded Above": False,
            "Monotonic": True,
            "Computational Cost": "Very Low",
            "Gradient Flow": "Good (but can die)",
            "Common Use": "CNNs, older transformers",
        },
        "GELU": {
            "Smooth": True,
            "Bounded Below": False,
            "Bounded Above": False,
            "Monotonic": False,
            "Computational Cost": "Medium",
            "Gradient Flow": "Excellent",
            "Common Use": "BERT, GPT-2/3, modern transformers",
        },
        "SiLU": {
            "Smooth": True,
            "Bounded Below": True,
            "Bounded Above": False,
            "Monotonic": False,
            "Computational Cost": "Low",
            "Gradient Flow": "Excellent",
            "Common Use": "LLaMA, modern LLMs",
        },
    }

    print("\nActivation Function Properties Comparison:")
    print("=" * 80)

    # Print header
    props = list(next(iter(properties.values())).keys())
    print(f"{'Property':<25} {'ReLU':<15} {'GELU':<15} {'SiLU':<15}")
    print("-" * 80)

    # Print each property
    for prop in props:
        values = [str(properties[act][prop]) for act in ["ReLU", "GELU", "SiLU"]]
        print(f"{prop:<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

    print("=" * 80)


# Test function for students
def test_activations():
    """
    Test activation functions with sample inputs.

    This function helps students verify their implementations.
    """
    # Test inputs
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("Testing Activation Functions:")
    print(f"Input: {x.tolist()}\n")

    # Test ReLU
    relu = ReLU()
    print(f"ReLU output: {relu(x).tolist()}")

    # Test GELU
    gelu = GELU()
    print(f"GELU output: {gelu(x).tolist()}")

    # Test SiLU
    silu = SiLU()
    print(f"SiLU output: {silu(x).tolist()}")

    # Test GLU
    glu = GLU()
    x_glu = torch.randn(2, 10)  # Will be split into 2x5
    print(f"\nGLU input shape: {x_glu.shape}")
    print(f"GLU output shape: {glu(x_glu).shape}")
