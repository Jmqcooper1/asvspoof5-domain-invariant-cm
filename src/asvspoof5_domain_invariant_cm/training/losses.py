"""Loss function utilities for training."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 2,
    method: str = "balanced",
) -> torch.Tensor:
    """Compute class weights for handling imbalanced data.

    Args:
        labels: Array of integer labels.
        num_classes: Number of classes.
        method: Weighting method ('balanced', 'inverse', 'sqrt_inverse').

    Returns:
        Tensor of class weights.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    if method == "balanced":
        # sklearn balanced: n_samples / (n_classes * n_samples_per_class)
        weights = total / (num_classes * counts.clip(min=1))
    elif method == "inverse":
        weights = total / counts.clip(min=1)
    elif method == "sqrt_inverse":
        weights = np.sqrt(total / counts.clip(min=1))
    else:
        weights = np.ones(num_classes)

    # Normalize
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)


class TaskLoss(nn.Module):
    """Task loss (bonafide/spoof classification).

    Args:
        label_smoothing: Label smoothing factor.
        class_weights: Optional class weights tensor.
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(logits, labels)


class DomainLoss(nn.Module):
    """Domain classification loss.

    Args:
        num_classes: Number of domain classes.
        class_weights: Optional class weights.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(logits, labels)


class CombinedDANNLoss(nn.Module):
    """Combined loss for DANN training.

    L_total = L_task + lambda * (L_codec + L_codec_q)

    Args:
        task_label_smoothing: Label smoothing for task loss.
        task_class_weights: Optional class weights for task loss.
        lambda_domain: Weight for domain losses.
    """

    def __init__(
        self,
        task_label_smoothing: float = 0.0,
        task_class_weights: Optional[torch.Tensor] = None,
        lambda_domain: float = 0.1,
    ):
        super().__init__()
        self.task_loss = TaskLoss(
            label_smoothing=task_label_smoothing,
            class_weights=task_class_weights,
        )
        self.codec_loss = DomainLoss(num_classes=0)  # num_classes not needed
        self.codec_q_loss = DomainLoss(num_classes=0)
        self.lambda_domain = lambda_domain

    def forward(
        self,
        task_logits: torch.Tensor,
        codec_logits: torch.Tensor,
        codec_q_logits: torch.Tensor,
        task_labels: torch.Tensor,
        codec_labels: torch.Tensor,
        codec_q_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined DANN loss.

        Returns:
            Dictionary with total_loss and component losses.
        """
        l_task = self.task_loss(task_logits, task_labels)
        l_codec = self.codec_loss(codec_logits, codec_labels)
        l_codec_q = self.codec_q_loss(codec_q_logits, codec_q_labels)

        # GRL already negates gradients, so we add (not subtract)
        total_loss = l_task + self.lambda_domain * (l_codec + l_codec_q)

        return {
            "total_loss": total_loss,
            "task_loss": l_task,
            "codec_loss": l_codec,
            "codec_q_loss": l_codec_q,
        }

    def set_lambda(self, lambda_domain: float) -> None:
        """Update domain loss weight."""
        self.lambda_domain = lambda_domain


class CombinedERMLoss(nn.Module):
    """Combined loss for ERM training (task loss only).

    Args:
        task_label_smoothing: Label smoothing for task loss.
        task_class_weights: Optional class weights for task loss.
    """

    def __init__(
        self,
        task_label_smoothing: float = 0.0,
        task_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.task_loss = TaskLoss(
            label_smoothing=task_label_smoothing,
            class_weights=task_class_weights,
        )

    def forward(
        self,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute ERM loss.

        Returns:
            Dictionary with total_loss and task_loss.
        """
        l_task = self.task_loss(task_logits, task_labels)

        return {
            "total_loss": l_task,
            "task_loss": l_task,
        }


def build_loss(
    method: str,
    task_label_smoothing: float = 0.0,
    task_class_weights: Optional[torch.Tensor] = None,
    lambda_domain: float = 0.1,
) -> nn.Module:
    """Build loss function from config.

    Args:
        method: Training method ('erm' or 'dann').
        task_label_smoothing: Label smoothing.
        task_class_weights: Class weights for task loss.
        lambda_domain: Domain loss weight (for DANN).

    Returns:
        Loss module.
    """
    if method == "dann":
        return CombinedDANNLoss(
            task_label_smoothing=task_label_smoothing,
            task_class_weights=task_class_weights,
            lambda_domain=lambda_domain,
        )
    else:
        return CombinedERMLoss(
            task_label_smoothing=task_label_smoothing,
            task_class_weights=task_class_weights,
        )
