"""
Transformer Language Model

Required baseline scope in this file:
- token embeddings
- one positional-encoding path
- decoder stack
- final LM head
- forward pass for next-token prediction

Optional extension in this file:
- RoPE integration
- extra architecture variants beyond the baseline path
- more advanced generation helpers
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEncoding,
    TransformerDecoderLayer,
    create_causal_mask,
    create_padding_mask,
)
from .config import ModelConfig


class TransformerLanguageModel(nn.Module):
    """
    Transformer-based Language Model.

    This is a decoder-only transformer model for autoregressive language modeling.
    It predicts the next token given the previous tokens.

    Architecture:
    1. Token Embedding
    2. Positional Encoding
    3. Stack of Transformer Decoder Layers
    4. Output Layer (projects to vocabulary)

    Args:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # TODO: Create token embedding layer
        # This converts token IDs to dense vectors
        # Hint: Use nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.token_embedding = None  # STUDENT TODO

        # TODO: Create positional encoding based on config
        # Hint: Check config.pos_encoding_type and create the appropriate module
        if config.pos_encoding_type == "sinusoidal":
            self.pos_encoding = None  # STUDENT TODO
        elif config.pos_encoding_type == "rope":
            # RoPE is applied in attention, so we just store it
            self.pos_encoding = None  # STUDENT TODO (use head_dim)
        elif config.pos_encoding_type == "learned":
            self.pos_encoding = None  # STUDENT TODO
        else:
            raise ValueError(f"Unknown pos_encoding_type: {config.pos_encoding_type}")

        # TODO: Create stack of transformer decoder layers
        # Hint: Use nn.ModuleList with TransformerDecoderLayer for each layer
        self.layers = None  # STUDENT TODO

        # TODO: Create final layer normalization
        # Hint: Use LayerNorm or RMSNorm based on config.norm_type
        if config.norm_type == "layernorm":
            from ..components import LayerNorm
            self.final_norm = None  # STUDENT TODO
        elif config.norm_type == "rmsnorm":
            from ..components import RMSNorm
            self.final_norm = None  # STUDENT TODO

        # TODO: Create output projection layer (language modeling head)
        # This projects from d_model to vocab_size
        # Hint: Use nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head = None  # STUDENT TODO

        # TODO: Tie input and output embeddings if specified
        # This shares weights between token embedding and output projection
        # It reduces parameters and often improves performance
        if config.tie_word_embeddings:
            # STUDENT TODO: Set self.lm_head.weight = self.token_embedding.weight
            pass

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights using Xavier/Glorot initialization.

        This is called automatically for all modules via self.apply().
        """
        if isinstance(module, nn.Linear):
            # TODO: Initialize linear layer weights
            # Hint: Use torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            pass  # STUDENT TODO
            if module.bias is not None:
                # TODO: Initialize bias to zero
                # Hint: Use torch.nn.init.zeros_(module.bias)
                pass  # STUDENT TODO
        elif isinstance(module, nn.Embedding):
            # TODO: Initialize embedding weights
            # Hint: Use torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            pass  # STUDENT TODO

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass of the language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
                           1 for tokens to attend to, 0 for padding tokens
            return_hidden_states: Whether to return hidden states from all layers

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden_states: Optional list of hidden states from each layer
        """
        batch_size, seq_len = input_ids.shape

        # TODO: Get token embeddings
        # Hint: self.token_embedding(input_ids)
        # Shape: (batch_size, seq_len, d_model)
        x = None  # STUDENT TODO

        # TODO: Apply positional encoding
        # The method depends on the type of positional encoding
        if self.config.pos_encoding_type in ["sinusoidal", "learned"]:
            # For sinusoidal and learned, we add the encoding to embeddings
            # Hint: x = self.pos_encoding(x)
            x = None  # STUDENT TODO
        # For RoPE, we'll apply it inside the attention mechanism

        # TODO: Create causal mask for autoregressive generation
        # Hint: Use create_causal_mask(seq_len, input_ids.device)
        causal_mask = None  # STUDENT TODO

        # TODO: Combine causal mask with padding mask if provided
        if attention_mask is not None:
            # Create padding mask
            # Hint: Use create_padding_mask(input_ids, self.config.pad_token_id)
            # Or use the provided attention_mask
            padding_mask = None  # STUDENT TODO

            # Combine masks: both causal and padding must be satisfied
            # Hint: Use element-wise multiplication or logical AND
            # Shape: (batch_size, 1, seq_len, seq_len)
            mask = None  # STUDENT TODO
        else:
            mask = causal_mask

        # Store hidden states if requested
        hidden_states = [] if return_hidden_states else None

        # TODO: Pass through transformer layers
        for layer in self.layers:
            if return_hidden_states:
                hidden_states.append(x)

            # Apply RoPE if using rotary positional encoding
            if self.config.pos_encoding_type == "rope":
                # RoPE is applied inside the attention mechanism
                # We need to pass it to the layer somehow
                # For simplicity, we'll modify the layer to handle RoPE internally
                # STUDENT TODO: Think about how to integrate RoPE with attention
                pass

            # TODO: Apply transformer layer
            # Hint: x = layer(x, self_attn_mask=mask)
            x = None  # STUDENT TODO

        # TODO: Apply final layer normalization
        # Hint: x = self.final_norm(x)
        x = None  # STUDENT TODO

        # TODO: Project to vocabulary
        # Hint: logits = self.lm_head(x)
        # Shape: (batch_size, seq_len, vocab_size)
        logits = None  # STUDENT TODO

        return logits, hidden_states

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        This is a simple generation method. Students will implement more sophisticated
        generation strategies in the generation module.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from top p probability mass
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # TODO: Get logits for the last token
                # Hint: Forward pass, then take logits[:, -1, :]
                logits, _ = None, None  # STUDENT TODO
                logits = None  # STUDENT TODO (select last token)

                # TODO: Apply temperature
                # Hint: logits = logits / temperature
                logits = None  # STUDENT TODO

                # TODO: Apply top-k filtering if specified
                if top_k is not None:
                    # Keep only top k logits, set others to -inf
                    # Hint: Use torch.topk()
                    pass  # STUDENT TODO

                # TODO: Apply top-p (nucleus) filtering if specified
                if top_p is not None:
                    # Keep only tokens with cumulative probability <= top_p
                    # This is more complex, students will implement in generation module
                    pass  # STUDENT TODO

                # TODO: Sample next token
                if do_sample:
                    # Sample from the distribution
                    # Hint: Use F.softmax() and torch.multinomial()
                    probs = None  # STUDENT TODO
                    next_token = None  # STUDENT TODO
                else:
                    # Greedy decoding: take the most likely token
                    # Hint: Use torch.argmax()
                    next_token = None  # STUDENT TODO

                # TODO: Append next token to generated sequence
                # Hint: Use torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
                generated = None  # STUDENT TODO

                # TODO: Check for EOS token
                # If all sequences have generated EOS, stop
                if (next_token == self.config.eos_token_id).all():
                    break

        return generated

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estimate model flops utilization (MFU).

        This helps students understand the computational efficiency of their model.

        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time per iteration in seconds

        Returns:
            MFU as a fraction of peak FLOPS
        """
        # Estimate FLOPs per token
        # This is a rough estimate based on the model architecture
        N = self.get_num_params()
        L, H, Q, T = (
            self.config.num_layers,
            self.config.num_heads,
            self.config.d_model // self.config.num_heads,
            self.config.max_seq_len,
        )

        # Approximate FLOPs per token (forward pass)
        # Attention: 2 * 2 * L * H * Q * T (QK^T and attention * V)
        # FFN: 2 * 2 * L * d_model * d_ff
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T * fwdbwd_per_iter * 3  # 3x for backward
        flops_per_iter = flops_per_fwdbwd
        flops_achieved = flops_per_iter / dt

        # Get peak FLOPS for the device (rough estimate)
        # A100: ~312 TFLOPS (fp16), V100: ~125 TFLOPS (fp16), T4: ~65 TFLOPS (fp16)
        # For CPU or unknown GPU, use a conservative estimate
        flops_promised = 65e12  # 65 TFLOPS (T4 GPU)

        mfu = flops_achieved / flops_promised
        return mfu


# Test function
def test_language_model():
    """
    Test the language model implementation.

    This function helps students verify their implementation.
    """
    from .config import get_small_config

    config = get_small_config()
    model = TransformerLanguageModel(config)

    print("Testing Transformer Language Model:")
    print(f"Configuration: {config.to_dict()}\n")

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"Input shape: {input_ids.shape}")

    logits, hidden_states = model(input_ids, return_hidden_states=True)
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of hidden states: {len(hidden_states)}")

    # Test generation
    print("\nTesting generation:")
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")

    # Model statistics
    print(f"\nModel statistics:")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True):,}")
