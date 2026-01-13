"""Learning rate and DANN lambda schedulers."""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create linear schedule with warmup.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        last_epoch: Last epoch (for resume).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create cosine schedule with warmup.

    Args:
        optimizer: Optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total training steps.
        num_cycles: Number of cosine cycles.
        min_lr_ratio: Minimum LR as ratio of initial LR.
        last_epoch: Last epoch (for resume).

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class LambdaScheduler:
    """DANN lambda scheduler for GRL strength.

    Args:
        schedule_type: Type of schedule ('constant', 'linear', 'exponential').
        start_value: Starting lambda value.
        end_value: Ending lambda value.
        warmup_epochs: Number of warmup epochs (lambda=0).
        total_epochs: Total training epochs.
    """

    def __init__(
        self,
        schedule_type: str = "constant",
        start_value: float = 0.1,
        end_value: float = 1.0,
        warmup_epochs: int = 0,
        total_epochs: int = 50,
    ):
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_lambda(self, epoch: int) -> float:
        """Get lambda value for current epoch.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            Lambda value.
        """
        if epoch < self.warmup_epochs:
            return 0.0

        if self.schedule_type == "constant":
            return self.start_value

        # Adjust epoch for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        effective_epochs = self.total_epochs - self.warmup_epochs

        if effective_epochs <= 0:
            return self.end_value

        progress = adjusted_epoch / effective_epochs

        if self.schedule_type == "linear":
            return self.start_value + (self.end_value - self.start_value) * progress
        elif self.schedule_type == "exponential":
            # Exponential ramp-up: lambda = 2 / (1 + exp(-10 * p)) - 1
            return (
                2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
            ) * (self.end_value - self.start_value) + self.start_value
        else:
            return self.start_value


def build_optimizer(
    model: torch.nn.Module,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """Build optimizer from config.

    Args:
        model: Model to optimize.
        name: Optimizer name ('adam', 'adamw', 'sgd').
        lr: Learning rate.
        weight_decay: Weight decay.
        betas: Adam betas.

    Returns:
        Optimizer instance.
    """
    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    if name.lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name.lower() == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif name.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_lr_scheduler(
    optimizer: Optimizer,
    name: str = "cosine",
    num_warmup_steps: int = 500,
    num_training_steps: int = 10000,
    min_lr_ratio: float = 0.01,
) -> Optional[LambdaLR]:
    """Build learning rate scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        name: Scheduler name ('cosine', 'linear', 'constant').
        num_warmup_steps: Warmup steps.
        num_training_steps: Total training steps.
        min_lr_ratio: Minimum LR ratio (for cosine).

    Returns:
        Scheduler instance or None.
    """
    if name.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
    elif name.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif name.lower() == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
