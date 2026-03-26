"""
Text Generation Module

Required baseline scope in this file:
- greedy decoding
- temperature sampling
- at least one filtering strategy (`top-k` or `top-p`)

Optional extension in this file:
- beam search
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGenerator:
    """
    Text generator with multiple sampling strategies.

    This class provides various methods for generating text from a trained language model.
    For the baseline path, prioritize greedy decoding plus one stochastic method.

    Args:
        model: Trained language model
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run generation on
    """

    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from top p probability mass
            do_sample: Whether to sample or use greedy decoding
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated text strings
        """
        # TODO: Encode prompt
        # STUDENT TODO: Use tokenizer to encode prompt
        input_ids = None  # STUDENT TODO: self.tokenizer.encode(prompt, add_special_tokens=True)

        # TODO: Convert to tensor and move to device
        # STUDENT TODO: Create tensor and move to device
        input_ids = None  # STUDENT TODO: torch.tensor([input_ids], device=self.device)

        # TODO: Repeat for multiple sequences
        if num_return_sequences > 1:
            # STUDENT TODO: Repeat input_ids for num_return_sequences
            # Hint: input_ids.repeat(num_return_sequences, 1)
            input_ids = None  # STUDENT TODO

        # TODO: Generate tokens
        # STUDENT TODO: Call _generate_tokens
        generated_ids = None  # STUDENT TODO

        # TODO: Decode generated sequences
        generated_texts = []
        for ids in generated_ids:
            # STUDENT TODO: Decode token IDs to text
            text = None  # STUDENT TODO: self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            do_sample: Whether to sample

        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        generated = input_ids

        for _ in range(max_new_tokens):
            # TODO: Get logits for next token
            # STUDENT TODO: Forward pass through model
            # Note: Only need logits for the last position
            logits, _ = None, None  # STUDENT TODO: self.model(generated)
            logits = None  # STUDENT TODO: logits[:, -1, :] (select last position)

            # TODO: Apply temperature
            # STUDENT TODO: Divide logits by temperature
            if temperature != 1.0:
                logits = None  # STUDENT TODO

            # TODO: Apply top-k filtering
            if top_k is not None:
                # STUDENT TODO: Keep only top-k logits, set others to -inf
                logits = self._top_k_filtering(logits, top_k)

            # TODO: Apply top-p filtering
            if top_p is not None:
                # STUDENT TODO: Keep only top-p probability mass
                logits = self._top_p_filtering(logits, top_p)

            # TODO: Sample or select next token
            if do_sample:
                # STUDENT TODO: Sample from distribution
                # Hint: F.softmax(logits, dim=-1) then torch.multinomial
                probs = None  # STUDENT TODO
                next_token = None  # STUDENT TODO
            else:
                # STUDENT TODO: Greedy decoding - take argmax
                next_token = None  # STUDENT TODO

            # TODO: Append next token to generated sequence
            # STUDENT TODO: Concatenate next_token to generated
            # Hint: torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            generated = None  # STUDENT TODO

            # TODO: Check for EOS token
            # If all sequences have generated EOS, stop
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        return generated

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Filter logits to keep only top-k tokens.

        Args:
            logits: Logits of shape (batch_size, vocab_size)
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits with non-top-k tokens set to -inf
        """
        # TODO: Get top-k values and indices
        # STUDENT TODO: Use torch.topk to get top-k values
        top_k_values, top_k_indices = None, None  # STUDENT TODO

        # TODO: Create mask for top-k tokens
        # STUDENT TODO: Set all logits to -inf, then restore top-k values
        # Hint: Create tensor of -inf, then scatter top-k values
        filtered_logits = torch.full_like(logits, float("-inf"))
        # STUDENT TODO: Use scatter_ to restore top-k values
        # Hint: filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        pass  # STUDENT TODO

        return filtered_logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.

        Keep only tokens with cumulative probability <= top_p.

        Args:
            logits: Logits of shape (batch_size, vocab_size)
            top_p: Cumulative probability threshold

        Returns:
            Filtered logits
        """
        # TODO: Sort logits in descending order
        # STUDENT TODO: Sort logits and get indices
        sorted_logits, sorted_indices = None, None  # STUDENT TODO: torch.sort(logits, descending=True, dim=-1)

        # TODO: Compute cumulative probabilities
        # STUDENT TODO: Apply softmax and compute cumsum
        sorted_probs = None  # STUDENT TODO: F.softmax(sorted_logits, dim=-1)
        cumulative_probs = None  # STUDENT TODO: torch.cumsum(sorted_probs, dim=-1)

        # TODO: Find tokens to remove (cumulative prob > top_p)
        # STUDENT TODO: Create mask for tokens to remove
        # Shift right to keep at least one token
        sorted_indices_to_remove = cumulative_probs > top_p
        # STUDENT TODO: Shift mask to the right to keep first token above threshold
        # Hint: sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        # Hint: sorted_indices_to_remove[..., 0] = False
        pass  # STUDENT TODO

        # TODO: Set removed tokens to -inf
        # STUDENT TODO: Scatter -inf values back to original positions
        # Hint: Create filtered_logits, then use scatter_ with sorted_indices
        filtered_logits = logits.clone()
        # STUDENT TODO: Implement scattering
        pass  # STUDENT TODO

        return filtered_logits

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        # TODO: Generate for each prompt
        # STUDENT TODO: Loop through prompts and generate
        # Note: This is a simple implementation. For efficiency, you'd want to
        # batch the prompts together, but that requires padding handling.
        results = []
        for prompt in prompts:
            # STUDENT TODO: Generate for this prompt
            generated = None  # STUDENT TODO: self.generate(prompt, max_new_tokens, **kwargs)
            results.extend(generated)

        return results


class BeamSearchGenerator:
    """
    OPTIONAL EXTENSION.

    Beam search text generator.

    Beam search maintains multiple hypotheses and selects the most likely sequence.
    This is more advanced and optional for students.

    Args:
        model: Trained language model
        tokenizer: Tokenizer
        device: Device to run on
        beam_width: Number of beams to maintain
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        beam_width: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beam_width = beam_width
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Generate text using beam search.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            length_penalty: Length penalty (> 1.0 favors longer sequences)

        Returns:
            Generated text
        """
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)

        # Each beam is (sequence, score)
        beams = [(input_ids, 0.0)]

        for _ in range(max_new_tokens):
            candidates = []

            for seq, score in beams:
                logits, _ = self.model(seq)
                logits = logits[:, -1, :]

                log_probs = F.log_softmax(logits, dim=-1)

                top_log_probs, top_indices = torch.topk(log_probs, self.beam_width, dim=-1)

                for i in range(self.beam_width):
                    next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    next_seq = torch.cat([seq, next_token], dim=-1)
                    next_score = score + top_log_probs[0, i].item()

                    # Apply length penalty
                    length_normalized_score = next_score / (next_seq.size(1) ** length_penalty)

                    candidates.append((next_seq, next_score, length_normalized_score))

            candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in candidates[: self.beam_width]]

            if all(seq[0, -1].item() == self.tokenizer.eos_token_id for seq, _ in beams):
                break

        best_seq, _ = beams[0]
        generated_text = self.tokenizer.decode(best_seq[0].tolist(), skip_special_tokens=True)

        return generated_text


# Utility functions
def compare_sampling_strategies(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
):
    """
    Compare different sampling strategies on the same prompt.

    This helps students understand the differences between strategies.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input prompt
        device: Device
        max_new_tokens: Max tokens to generate
    """
    generator = TextGenerator(model, tokenizer, device)

    strategies = {
        "Greedy": {"do_sample": False},
        "Temperature 0.7": {"temperature": 0.7, "do_sample": True},
        "Temperature 1.5": {"temperature": 1.5, "do_sample": True},
        "Top-k (k=50)": {"top_k": 50, "do_sample": True},
        "Top-p (p=0.9)": {"top_p": 0.9, "do_sample": True},
        "Top-k + Top-p": {"top_k": 50, "top_p": 0.9, "do_sample": True},
    }

    print(f"Prompt: {prompt}\n")
    print("=" * 80)

    for name, kwargs in strategies.items():
        generated = generator.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        print(f"\n{name}:")
        print(f"{generated[0]}")
        print("-" * 80)


# Test function
def test_generator():
    """
    Test text generator.
    """
    from ..model.config import get_small_config
    from ..model.language_model import TransformerLanguageModel
    from ..tokenizer.base import CharacterTokenizer

    # Create dummy model and tokenizer
    texts = ["Hello world, this is a test."]
    tokenizer = CharacterTokenizer()
    tokenizer.train(texts)

    config = get_small_config()
    config.vocab_size = tokenizer.vocab_size
    model = TransformerLanguageModel(config)

    device = torch.device("cpu")
    model = model.to(device)

    # Create generator
    generator = TextGenerator(model, tokenizer, device)

    print("Testing Text Generator:")
    prompt = "Hello"
    print(f"Prompt: {prompt}\n")

    # Test greedy
    print("1. Greedy decoding:")
    generated = generator.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"   {generated[0]}\n")

    # Test sampling
    print("2. Temperature sampling:")
    generated = generator.generate(prompt, max_new_tokens=10, temperature=0.8, do_sample=True)
    print(f"   {generated[0]}\n")

    # Test top-k
    print("3. Top-k sampling:")
    generated = generator.generate(prompt, max_new_tokens=10, top_k=5, do_sample=True)
    print(f"   {generated[0]}\n")

    print("Generator test complete!")
