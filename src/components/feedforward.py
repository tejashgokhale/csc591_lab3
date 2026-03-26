"""
Feed-Forward Networks for Transformer Models

Required baseline scope in this file:
1. Standard Position-wise Feed-Forward Network (FFN)
2. One working FFN path for the baseline, usually the standard FFN

Optional extension in this file:
3. Gated Linear Unit (GLU) variants such as SwiGLU / GeGLU
4. Simple Mixture of Experts (MoE)

Feed-forward networks are applied position-wise (independently to each position)
and provide the model with non-linear transformations.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation


class PositionWiseFeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network.

    This is the FFN used in the original Transformer paper. It consists of two
    linear transformations with an activation function in between.

    Formula:
        FFN(x) = activation(xW1 + b1)W2 + b2

    Structure:
        Linear(d_model -> d_ff) -> Activation -> Dropout -> Linear(d_ff -> d_model)

    Args:
        d_model: Dimension of the model (input and output dimension)
        d_ff: Dimension of the feed-forward layer (hidden dimension)
        dropout: Dropout probability
        activation: Activation function name ("relu", "gelu", "silu")
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        # TODO: Create first linear layer (d_model -> d_ff)
        # Hint: Use nn.Linear(d_model, d_ff)
        self.linear1 = None  # STUDENT TODO

        # TODO: Create second linear layer (d_ff -> d_model)
        self.linear2 = None  # STUDENT TODO

        # TODO: Get activation function
        # Hint: Use get_activation(activation)
        self.activation = None  # STUDENT TODO

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply position-wise feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Apply FFN: linear1 -> activation -> dropout -> linear2
        # Hint: Chain the operations: self.linear1(x), self.activation(...), etc.
        x = None  # STUDENT TODO (apply linear1)
        x = None  # STUDENT TODO (apply activation)
        x = None  # STUDENT TODO (apply dropout)
        x = None  # STUDENT TODO (apply linear2)

        return x


class GLUFeedForward(nn.Module):
    """
    Gated Linear Unit (GLU) Feed-Forward Network.

    GLU variants use a gating mechanism to control information flow. They have been
    shown to improve performance in language models.

    There are several variants:
    - GLU: gate with sigmoid activation
    - SwiGLU: gate with SiLU/Swish activation (used in LLaMA)
    - GeGLU: gate with GELU activation (used in some transformers)

    Formula:
        GLU(x) = (xW1 + b1) ⊗ activation(xW2 + b2)
    where ⊗ is element-wise multiplication.

    In practice, we can compute both projections together and split:
        GLU(x) = Linear(x)[:, :d_ff] ⊗ activation(Linear(x)[:, d_ff:])

    Structure:
        Linear(d_model -> 2*d_ff) -> Split -> [Identity, Activation] -> Multiply -> Linear(d_ff -> d_model)

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward layer
        dropout: Dropout probability
        activation: Activation function for gating ("sigmoid", "silu", "gelu")
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()

        # TODO: Create first linear layer (d_model -> 2*d_ff)
        # Note: We project to 2*d_ff because we'll split into two parts
        # Hint: Use nn.Linear(d_model, 2 * d_ff)
        self.linear1 = None  # STUDENT TODO

        # TODO: Create second linear layer (d_ff -> d_model)
        self.linear2 = None  # STUDENT TODO

        # TODO: Get activation function for gating
        self.activation = None  # STUDENT TODO

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GLU feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # TODO: Apply first linear layer
        x = None  # STUDENT TODO

        # TODO: Split into two parts along the last dimension
        # Hint: Use torch.chunk(x, 2, dim=-1)
        x1, x2 = None, None  # STUDENT TODO

        # TODO: Apply gating: x1 * activation(x2)
        # Hint: Element-wise multiplication
        x = None  # STUDENT TODO

        # TODO: Apply dropout and second linear layer
        x = None  # STUDENT TODO (dropout)
        x = None  # STUDENT TODO (linear2)

        return x


class MixtureOfExperts(nn.Module):
    """
    OPTIONAL EXTENSION.

    Simple Mixture of Experts (MoE) Feed-Forward Network.

    MoE uses multiple "expert" networks and a gating mechanism to route inputs
    to different experts. This allows the model to specialize different experts
    for different types of inputs.

    Structure:
    1. Gating network: Computes routing probabilities for each expert
    2. Expert networks: Multiple FFN networks
    3. Routing: Select top-k experts for each token
    4. Combination: Weighted sum of expert outputs

    This is a simplified version. Production MoE systems (like in Switch Transformer)
    have additional features like load balancing losses and more sophisticated routing.

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of each expert's feed-forward layer
        num_experts: Number of expert networks
        top_k: Number of experts to route each token to
        dropout: Dropout probability
        activation: Activation function for experts
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        # TODO: Create gating network (router)
        # This is a simple linear layer that outputs logits for each expert
        # Hint: Use nn.Linear(d_model, num_experts)
        self.gate = None  # STUDENT TODO

        # TODO: Create expert networks
        # Each expert is a standard FFN
        # Hint: Use nn.ModuleList with PositionWiseFeedForward for each expert
        self.experts = None  # STUDENT TODO

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply mixture of experts.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # TODO: Compute gating scores
        # Shape: (batch_size, seq_len, num_experts)
        # Hint: Apply self.gate(x)
        gate_logits = None  # STUDENT TODO

        # TODO: Get top-k experts for each token
        # Hint: Use torch.topk(gate_logits, self.top_k, dim=-1)
        # This returns (values, indices) where:
        #   values: top-k gate logits
        #   indices: indices of top-k experts
        gate_scores, expert_indices = None, None  # STUDENT TODO

        # TODO: Apply softmax to gate scores to get routing probabilities
        # Hint: Use F.softmax(gate_scores, dim=-1)
        gate_probs = None  # STUDENT TODO

        # Initialize output
        output = torch.zeros_like(x)

        # TODO: Route to experts and combine outputs
        # For each of the top-k experts:
        #   1. Get the expert index
        #   2. Apply the expert to the input
        #   3. Weight by the gate probability
        #   4. Add to output
        #
        # Hint: Loop over self.top_k
        # For each k:
        #   expert_idx = expert_indices[:, :, k]  # Shape: (batch, seq_len)
        #   expert_prob = gate_probs[:, :, k:k+1]  # Shape: (batch, seq_len, 1)
        #   Then apply the expert and weight by probability
        #
        # Note: This is a simplified implementation. In practice, you'd want to
        # batch the expert computations for efficiency.

        for k in range(self.top_k):
            # STUDENT TODO: Implement expert routing
            # Get expert indices for this k
            # Apply corresponding experts
            # Weight by gate probabilities
            # Add to output
            pass  # STUDENT TODO

        return output

    def load_balancing_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage uniform expert usage.

        This auxiliary loss encourages the model to use all experts equally,
        preventing the model from only using a few experts.

        Args:
            gate_logits: Gate logits of shape (batch_size, seq_len, num_experts)

        Returns:
            Scalar load balancing loss
        """
        # TODO: Compute the fraction of tokens routed to each expert
        # Hint: Use softmax on gate_logits, then average over batch and sequence
        # Shape: (num_experts,)
        expert_usage = None  # STUDENT TODO

        # TODO: Compute coefficient of variation (CV) as load balancing loss
        # CV = std / mean
        # We want to minimize this to encourage uniform distribution
        # Hint: Use torch.std() and torch.mean()
        loss = None  # STUDENT TODO

        return loss


# Factory function to create FFN based on type
def create_ffn(
    ffn_type: str,
    d_model: int,
    d_ff: int,
    dropout: float = 0.1,
    activation: str = "relu",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create feed-forward network of specified type.

    Args:
        ffn_type: Type of FFN ("standard", "glu", "swiglu", "geglu", "moe")
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        **kwargs: Additional arguments for specific FFN types (e.g., num_experts for MoE)

    Returns:
        Feed-forward network module
    """
    # TODO: Create and return the appropriate FFN based on ffn_type
    # Hint: Use if-elif-else or a dictionary
    if ffn_type == "standard":
        return None  # STUDENT TODO
    elif ffn_type == "glu":
        return None  # STUDENT TODO
    elif ffn_type == "swiglu":
        # SwiGLU uses SiLU activation
        return None  # STUDENT TODO
    elif ffn_type == "geglu":
        # GeGLU uses GELU activation
        return None  # STUDENT TODO
    elif ffn_type == "moe":
        return None  # STUDENT TODO
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")


# Test function
def test_ffn():
    """
    Test different FFN implementations.

    This function helps students verify their implementations.
    """
    batch_size, seq_len, d_model, d_ff = 2, 10, 64, 256

    x = torch.randn(batch_size, seq_len, d_model)

    print("Testing Feed-Forward Networks:")
    print(f"Input shape: {x.shape}\n")

    # Test standard FFN
    ffn_standard = PositionWiseFeedForward(d_model, d_ff)
    out_standard = ffn_standard(x)
    print(f"Standard FFN output shape: {out_standard.shape}")

    # Test GLU FFN
    ffn_glu = GLUFeedForward(d_model, d_ff)
    out_glu = ffn_glu(x)
    print(f"GLU FFN output shape: {out_glu.shape}")

    # Test MoE FFN
    ffn_moe = MixtureOfExperts(d_model, d_ff, num_experts=4, top_k=2)
    out_moe = ffn_moe(x)
    print(f"MoE FFN output shape: {out_moe.shape}")

    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    print(f"\nParameter counts:")
    print(f"Standard FFN: {count_params(ffn_standard):,}")
    print(f"GLU FFN: {count_params(ffn_glu):,}")
    print(f"MoE FFN: {count_params(ffn_moe):,}")
