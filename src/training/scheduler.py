"""
Learning Rate Schedulers

Baseline note:
- the schedulers in this file are now fully implemented for students
- students may read/modify them for experiments if they want
- these should not be a blocker for getting the training pipeline running
"""

import math
from typing import Optional

import torch.optim as optim


class _SchedulerMixin:
    """Small helper so trainer checkpoints can save/restore scheduler state."""

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict):
        self.current_step = state_dict.get("current_step", 0)


class WarmupScheduler(_SchedulerMixin):
    """
    Learning rate warmup scheduler.

    Warmup gradually increases the learning rate from 0 to the base learning rate
    over a specified number of steps. This helps stabilize training at the beginning.

    Formula:
        lr = base_lr * (current_step / warmup_steps)  for current_step < warmup_steps
        lr = base_lr                                   for current_step >= warmup_steps

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate (if None, uses optimizer's lr)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        base_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.warmup_steps <= 0:
            lr = self.base_lr
        elif self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class CosineAnnealingScheduler(_SchedulerMixin):
    """
    Cosine annealing learning rate scheduler.

    Decreases learning rate following a cosine curve from base_lr to min_lr.

    Formula:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * current_step / total_steps))

    This provides a smooth decay that often works better than linear or exponential decay.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        base_lr: Base learning rate (if None, uses optimizer's lr)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        # STUDENT TODO (conceptual, not code): make sure you can derive why the
        # cosine schedule uses progress in [0, 1] before applying cos(pi * progress).
        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class WarmupCosineScheduler(_SchedulerMixin):
    """
    Combined warmup and cosine annealing scheduler.

    This is the most commonly used scheduler in modern transformer training.
    It combines:
    1. Linear warmup for the first warmup_steps
    2. Cosine annealing for the remaining steps

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        base_lr: Base learning rate (if None, uses optimizer's lr)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            decay_steps = max(self.total_steps - self.warmup_steps, 1)
            progress = min(
                max(self.current_step - self.warmup_steps, 0) / decay_steps,
                1.0,
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class LinearScheduler(_SchedulerMixin):
    """
    Linear learning rate decay scheduler.

    Linearly decreases learning rate from base_lr to min_lr.

    Formula:
        lr = base_lr - (base_lr - min_lr) * (current_step / total_steps)

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        base_lr: Base learning rate (if None, uses optimizer's lr)
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        lr = self.base_lr - (self.base_lr - self.min_lr) * progress

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    **kwargs,
):
    """
    Factory function to create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("warmup", "cosine", "warmup_cosine", "linear")
        **kwargs: Additional arguments for the scheduler

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "warmup":
        return WarmupScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingScheduler(
            optimizer,
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 0),
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "linear":
        return LinearScheduler(
            optimizer,
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def visualize_scheduler(
    scheduler_type: str,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    save_path: str = "lr_schedule.png",
):
    """
    Visualize learning rate schedule.

    This helps students understand how different schedulers behave.

    Args:
        scheduler_type: Type of scheduler
        total_steps: Total training steps
        warmup_steps: Warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import torch.nn as nn

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    # Create scheduler
    if scheduler_type == "warmup":
        scheduler = WarmupScheduler(optimizer, warmup_steps, base_lr)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingScheduler(optimizer, total_steps, min_lr, base_lr)
    elif scheduler_type == "warmup_cosine":
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, min_lr, base_lr
        )
    elif scheduler_type == "linear":
        scheduler = LinearScheduler(optimizer, total_steps, min_lr, base_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # Collect learning rates
    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title(f"{scheduler_type.replace('_', ' ').title()} Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Learning rate schedule visualization saved to {save_path}")


def compare_schedulers(
    total_steps: int = 1000,
    warmup_steps: int = 100,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    save_path: str = "lr_schedules_comparison.png",
):
    """
    Compare different learning rate schedulers.

    Args:
        total_steps: Total training steps
        warmup_steps: Warmup steps
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    import torch.nn as nn

    schedulers = {
        "Warmup": WarmupScheduler,
        "Cosine": CosineAnnealingScheduler,
        "Warmup + Cosine": WarmupCosineScheduler,
        "Linear": LinearScheduler,
    }

    plt.figure(figsize=(12, 6))

    for name, scheduler_class in schedulers.items():
        # Create dummy model and optimizer
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

        # Create scheduler
        if name == "Warmup":
            scheduler = scheduler_class(optimizer, warmup_steps, base_lr)
        elif name == "Cosine":
            scheduler = scheduler_class(optimizer, total_steps, min_lr, base_lr)
        elif name == "Warmup + Cosine":
            scheduler = scheduler_class(
                optimizer, warmup_steps, total_steps, min_lr, base_lr
            )
        elif name == "Linear":
            scheduler = scheduler_class(optimizer, total_steps, min_lr, base_lr)

        # Collect learning rates
        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_lr())
            scheduler.step()

        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedulers Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Scheduler comparison saved to {save_path}")


# Test function
def test_schedulers():
    """
    Test learning rate schedulers.
    """
    import torch.nn as nn

    # Create dummy model and optimizer
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Testing Learning Rate Schedulers:\n")

    # Test warmup scheduler
    print("1. Warmup Scheduler:")
    scheduler = WarmupScheduler(optimizer, warmup_steps=10)
    for step in range(15):
        print(f"   Step {step}: lr = {scheduler.get_lr():.6f}")
        scheduler.step()

    # Reset optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Test cosine scheduler
    print("\n2. Cosine Annealing Scheduler:")
    scheduler = CosineAnnealingScheduler(optimizer, total_steps=20, min_lr=1e-5)
    for step in range(0, 20, 5):
        print(f"   Step {step}: lr = {scheduler.get_lr():.6f}")
        scheduler.step()

    print("\nVisualize schedulers with visualize_scheduler() and compare_schedulers()")
