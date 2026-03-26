"""
Loss Functions and Metrics for Language Modeling

This module implements loss functions and evaluation metrics for
autoregressive language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelingLoss(nn.Module):
    """
    Cross-entropy loss for language modeling with proper handling of padding.

    In language modeling, we predict the next token given previous tokens.
    We use cross-entropy loss between predicted and target tokens.

    Important: We need to ignore padding tokens in the loss calculation.

    Args:
        pad_token_id: ID of the padding token (to ignore in loss)
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
    """

    def __init__(self, pad_token_id: int = 0, label_smoothing: float = 0.0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions of shape (batch_size, seq_len, vocab_size)
            targets: Target token IDs of shape (batch_size, seq_len)

        Returns:
            Scalar loss value
        """
        # TODO: Reshape logits and targets for cross-entropy
        # Cross-entropy expects:
        # - logits: (N, C) where N = batch_size * seq_len, C = vocab_size
        # - targets: (N,) where N = batch_size * seq_len

        # STUDENT TODO: Reshape logits to (batch_size * seq_len, vocab_size)
        # Hint: Use logits.view(-1, logits.size(-1))
        logits = None  # STUDENT TODO

        # STUDENT TODO: Reshape targets to (batch_size * seq_len,)
        # Hint: Use targets.view(-1)
        targets = None  # STUDENT TODO

        # TODO: Compute cross-entropy loss
        # Hint: Use F.cross_entropy with ignore_index=self.pad_token_id
        # This will automatically ignore padding tokens
        # Also use label_smoothing parameter
        # STUDENT TODO: Compute loss
        loss = None  # STUDENT TODO

        return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity is a common metric for language models. It measures how well
    the model predicts the next token. Lower perplexity is better.

    Formula: perplexity = exp(cross_entropy_loss)

    Interpretation:
    - Perplexity of 1: Perfect prediction (model is certain)
    - Perplexity of N: Model is as uncertain as if it had to choose uniformly from N tokens
    - Lower perplexity = better model

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    # TODO: Compute perplexity
    # Formula: exp(loss)
    # STUDENT TODO: Compute perplexity
    perplexity = None  # STUDENT TODO

    return perplexity


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: int = 0,
) -> float:
    """
    Compute token-level accuracy.

    This measures what fraction of tokens are predicted correctly.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        pad_token_id: Padding token ID to ignore

    Returns:
        Accuracy as a float between 0 and 1
    """
    # TODO: Get predicted token IDs
    # STUDENT TODO: Get the token with highest probability
    # Hint: Use torch.argmax(logits, dim=-1)
    predictions = None  # STUDENT TODO

    # TODO: Create mask for non-padding tokens
    # STUDENT TODO: Create mask where targets != pad_token_id
    mask = None  # STUDENT TODO

    # TODO: Compute accuracy only on non-padding tokens
    # STUDENT TODO: Count correct predictions and divide by total non-padding tokens
    # Hint: (predictions == targets) gives boolean tensor
    # Use mask to select only non-padding positions
    correct = None  # STUDENT TODO
    total = None  # STUDENT TODO
    accuracy = None  # STUDENT TODO

    return accuracy.item()


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
    pad_token_id: int = 0,
) -> float:
    """
    Compute top-k accuracy.

    This measures what fraction of times the correct token is in the top-k predictions.

    Args:
        logits: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        k: Number of top predictions to consider
        pad_token_id: Padding token ID to ignore

    Returns:
        Top-k accuracy as a float between 0 and 1
    """
    # TODO: Get top-k predictions
    # STUDENT TODO: Get indices of top-k logits
    # Hint: Use torch.topk(logits, k, dim=-1)
    # This returns (values, indices)
    _, top_k_preds = None, None  # STUDENT TODO

    # TODO: Check if target is in top-k
    # STUDENT TODO: Expand targets to compare with top_k_preds
    # Hint: targets.unsqueeze(-1) expands to (batch, seq_len, 1)
    # Then compare with top_k_preds: (targets.unsqueeze(-1) == top_k_preds)
    # Use .any(dim=-1) to check if target is in any of the k predictions
    in_top_k = None  # STUDENT TODO

    # TODO: Apply mask and compute accuracy
    mask = (targets != pad_token_id)
    correct = (in_top_k & mask).sum()
    total = mask.sum()
    accuracy = correct.float() / total.float()

    return accuracy.item()


class MetricsTracker:
    """
    Track and compute running averages of metrics during training.

    This helps students monitor training progress and understand model performance.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: Metric name and value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0

            # TODO: Update running sum and count
            # STUDENT TODO: Add value to running sum
            self.metrics[name] += None  # STUDENT TODO

            # STUDENT TODO: Increment count
            self.counts[name] += None  # STUDENT TODO

    def compute(self) -> dict:
        """
        Compute average of all metrics.

        Returns:
            Dictionary of metric names and their averages
        """
        # TODO: Compute averages
        # STUDENT TODO: Divide each metric sum by its count
        averages = {}
        for name in self.metrics:
            averages[name] = None  # STUDENT TODO

        return averages

    def get(self, name: str) -> float:
        """
        Get average of a specific metric.

        Args:
            name: Metric name

        Returns:
            Average value of the metric
        """
        if name not in self.metrics:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def __repr__(self) -> str:
        """String representation of metrics."""
        averages = self.compute()
        return ", ".join([f"{k}: {v:.4f}" for k, v in averages.items()])


# Utility function to compute all metrics
def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    pad_token_id: int = 0,
) -> dict:
    """
    Compute all metrics for a batch.

    Args:
        logits: Model predictions
        targets: Target token IDs
        loss: Computed loss value
        pad_token_id: Padding token ID

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "loss": loss.item(),
        "perplexity": compute_perplexity(loss.item()),
        "accuracy": compute_accuracy(logits, targets, pad_token_id),
        "top5_accuracy": compute_top_k_accuracy(logits, targets, k=5, pad_token_id=pad_token_id),
    }

    return metrics


# Test function
def test_loss_and_metrics():
    """
    Test loss functions and metrics.
    """
    batch_size, seq_len, vocab_size = 2, 10, 100
    pad_token_id = 0

    # Create dummy data
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Add some padding
    targets[:, -3:] = pad_token_id

    print("Testing Loss and Metrics:")
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}\n")

    # Test loss
    loss_fn = LanguageModelingLoss(pad_token_id=pad_token_id)
    loss = loss_fn(logits, targets)
    print(f"Loss: {loss.item():.4f}")

    # Test perplexity
    perplexity = compute_perplexity(loss.item())
    print(f"Perplexity: {perplexity:.4f}")

    # Test accuracy
    accuracy = compute_accuracy(logits, targets, pad_token_id)
    print(f"Accuracy: {accuracy:.4f}")

    # Test top-k accuracy
    top5_acc = compute_top_k_accuracy(logits, targets, k=5, pad_token_id=pad_token_id)
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    # Test metrics tracker
    print("\nTesting MetricsTracker:")
    tracker = MetricsTracker()
    for i in range(5):
        tracker.update(loss=loss.item(), accuracy=accuracy)
    print(f"Tracked metrics: {tracker}")
    print(f"Average loss: {tracker.get('loss'):.4f}")
