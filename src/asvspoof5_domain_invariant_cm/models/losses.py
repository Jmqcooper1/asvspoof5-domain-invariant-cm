"""Loss functions for ERM and DANN training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DANNLoss(nn.Module):
    """Combined loss for DANN training.

    L_total = L_task + lambda * (L_codec + L_codec_q)

    Args:
        lambda_: Weight for domain adversarial losses.
        label_smoothing: Label smoothing for task loss.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_ = lambda_
        self.task_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.codec_loss = nn.CrossEntropyLoss()
        self.codec_q_loss = nn.CrossEntropyLoss()

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

        Args:
            task_logits: Bonafide/spoof predictions (B, 2).
            codec_logits: CODEC predictions (B, num_codecs).
            codec_q_logits: CODEC_Q predictions (B, num_codec_qs).
            task_labels: Ground truth task labels (B,).
            codec_labels: Ground truth CODEC labels (B,).
            codec_q_labels: Ground truth CODEC_Q labels (B,).

        Returns:
            Dictionary with total_loss and component losses.
        """
        # Task loss
        l_task = self.task_loss(task_logits, task_labels)

        # Domain losses (after GRL, so these push for invariance)
        l_codec = self.codec_loss(codec_logits, codec_labels)
        l_codec_q = self.codec_q_loss(codec_q_logits, codec_q_labels)

        # Combined loss
        # Note: GRL already negates gradients, so we add (not subtract)
        total_loss = l_task + self.lambda_ * (l_codec + l_codec_q)

        return {
            "total_loss": total_loss,
            "task_loss": l_task,
            "codec_loss": l_codec,
            "codec_q_loss": l_codec_q,
        }

    def set_lambda(self, lambda_: float) -> None:
        """Update domain loss weight."""
        self.lambda_ = lambda_


class ERMLoss(nn.Module):
    """Standard ERM loss (task loss only).

    Args:
        label_smoothing: Label smoothing factor.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute ERM loss.

        Args:
            logits: Predictions (B, num_classes).
            labels: Ground truth (B,).

        Returns:
            Dictionary with loss.
        """
        loss = self.loss(logits, labels)
        return {"total_loss": loss, "task_loss": loss}


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Args:
        alpha: Class weight (for positive class).
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predictions (B, num_classes).
            labels: Ground truth (B,).

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
