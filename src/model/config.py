"""
Model Configuration

This module defines the configuration class for transformer language models.
It allows easy experimentation with different model architectures and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for Transformer Language Model.

    This configuration class allows students to easily experiment with different
    model architectures and hyperparameters.

    Attributes:
        # Vocabulary and embedding
        vocab_size: Size of the vocabulary
        d_model: Dimension of the model (embedding dimension)
        max_seq_len: Maximum sequence length

        # Transformer architecture
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward network
        dropout: Dropout probability

        # Component choices
        attention_type: Type of attention ("mha" or "gqa")
        num_kv_heads: Number of key-value heads (for GQA, None means use num_heads)
        pos_encoding_type: Type of positional encoding ("sinusoidal", "rope", "learned")
        norm_type: Type of normalization ("layernorm" or "rmsnorm")
        norm_position: Position of normalization ("pre" or "post")
        ffn_type: Type of feed-forward network ("standard", "glu", "swiglu", "geglu", "moe")
        activation: Activation function ("relu", "gelu", "silu")

        # MoE specific (if ffn_type == "moe")
        num_experts: Number of experts in MoE
        top_k_experts: Number of experts to route to

        # Special tokens
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID

        # Training
        tie_word_embeddings: Whether to tie input and output embeddings
    """

    # Vocabulary and embedding
    vocab_size: int = 8000
    d_model: int = 512
    max_seq_len: int = 512

    # Transformer architecture
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

    # Component choices
    attention_type: str = "mha"  # "mha" or "gqa"
    num_kv_heads: Optional[int] = None  # For GQA, None means use num_heads
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal", "rope", or "learned"
    norm_type: str = "layernorm"  # "layernorm" or "rmsnorm"
    norm_position: str = "pre"  # "pre" or "post"
    ffn_type: str = "standard"  # "standard", "glu", "swiglu", "geglu", "moe"
    activation: str = "relu"  # "relu", "gelu", "silu"

    # MoE specific
    num_experts: int = 8
    top_k_experts: int = 2

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Training
    tie_word_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate d_model is divisible by num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")

        # Validate GQA configuration
        if self.attention_type == "gqa":
            if self.num_kv_heads is None:
                self.num_kv_heads = self.num_heads // 2  # Default to half
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
                )

        # Validate choices
        valid_attention_types = ["mha", "gqa"]
        if self.attention_type not in valid_attention_types:
            raise ValueError(f"attention_type must be one of {valid_attention_types}")

        valid_pos_encoding_types = ["sinusoidal", "rope", "learned"]
        if self.pos_encoding_type not in valid_pos_encoding_types:
            raise ValueError(f"pos_encoding_type must be one of {valid_pos_encoding_types}")

        valid_norm_types = ["layernorm", "rmsnorm"]
        if self.norm_type not in valid_norm_types:
            raise ValueError(f"norm_type must be one of {valid_norm_types}")

        valid_norm_positions = ["pre", "post"]
        if self.norm_position not in valid_norm_positions:
            raise ValueError(f"norm_position must be one of {valid_norm_positions}")

        valid_ffn_types = ["standard", "glu", "swiglu", "geglu", "moe"]
        if self.ffn_type not in valid_ffn_types:
            raise ValueError(f"ffn_type must be one of {valid_ffn_types}")

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.num_heads

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "attention_type": self.attention_type,
            "num_kv_heads": self.num_kv_heads,
            "pos_encoding_type": self.pos_encoding_type,
            "norm_type": self.norm_type,
            "norm_position": self.norm_position,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "num_experts": self.num_experts,
            "top_k_experts": self.top_k_experts,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, path: str):
        """Save configuration to file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load configuration from file."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Predefined configurations for different model sizes
def get_small_config() -> ModelConfig:
    """
    Small model configuration for limited hardware (e.g., Google Colab free tier).

    This configuration is designed to be trainable on limited resources while
    still demonstrating all the key concepts.
    """
    return ModelConfig(
        vocab_size=8000,
        d_model=256,
        max_seq_len=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        dropout=0.1,
        attention_type="mha",
        pos_encoding_type="sinusoidal",
        norm_type="layernorm",
        norm_position="pre",
        ffn_type="standard",
        activation="gelu",
    )


def get_medium_config() -> ModelConfig:
    """
    Medium model configuration for moderate hardware.
    """
    return ModelConfig(
        vocab_size=16000,
        d_model=512,
        max_seq_len=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        attention_type="gqa",
        num_kv_heads=4,
        pos_encoding_type="rope",
        norm_type="rmsnorm",
        norm_position="pre",
        ffn_type="swiglu",
        activation="silu",
    )


def get_large_config() -> ModelConfig:
    """
    Large model configuration (for reference, may not be trainable on limited hardware).
    """
    return ModelConfig(
        vocab_size=32000,
        d_model=768,
        max_seq_len=1024,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        dropout=0.1,
        attention_type="gqa",
        num_kv_heads=4,
        pos_encoding_type="rope",
        norm_type="rmsnorm",
        norm_position="pre",
        ffn_type="swiglu",
        activation="silu",
    )


# Configuration for experiments
def get_experiment_configs() -> dict:
    """
    Get configurations for different experiments.

    Returns a dictionary of configurations for comparing different components.
    """
    base_config = get_small_config()

    return {
        # Normalization comparison
        "layernorm": base_config,
        "rmsnorm": ModelConfig(**{**base_config.to_dict(), "norm_type": "rmsnorm"}),
        # Positional encoding comparison
        "sinusoidal": base_config,
        "rope": ModelConfig(**{**base_config.to_dict(), "pos_encoding_type": "rope"}),
        # Attention comparison
        "mha": base_config,
        "gqa": ModelConfig(**{**base_config.to_dict(), "attention_type": "gqa", "num_kv_heads": 2}),
        # FFN comparison
        "standard_ffn": base_config,
        "glu_ffn": ModelConfig(**{**base_config.to_dict(), "ffn_type": "swiglu", "activation": "silu"}),
        # Norm position comparison
        "pre_norm": base_config,
        "post_norm": ModelConfig(**{**base_config.to_dict(), "norm_position": "post"}),
    }
