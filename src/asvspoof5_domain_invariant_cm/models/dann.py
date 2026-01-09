"""Domain-Adversarial Neural Network components.

Implements:
- Gradient Reversal Layer (GRL)
- Multi-head domain discriminator for CODEC and CODEC_Q
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer function.

    Forward: identity
    Backward: negate gradients by lambda
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain-adversarial training.

    During forward pass: identity function
    During backward pass: negate gradients by factor lambda

    Args:
        lambda_: Gradient reversal scaling factor.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update lambda value."""
        self.lambda_ = lambda_


class MultiHeadDomainDiscriminator(nn.Module):
    """Multi-head domain discriminator for CODEC and CODEC_Q.

    Args:
        input_dim: Input feature dimension.
        num_codecs: Number of unique CODEC values.
        num_codec_qs: Number of unique CODEC_Q values.
        hidden_dim: Shared hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_codecs: int,
        num_codec_qs: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for CODEC and CODEC_Q
        self.codec_head = nn.Linear(hidden_dim, num_codecs)
        self.codec_q_head = nn.Linear(hidden_dim, num_codec_qs)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict domain labels.

        Args:
            x: Input features of shape (B, D).

        Returns:
            Tuple of (codec_logits, codec_q_logits).
        """
        shared = self.shared(x)
        codec_logits = self.codec_head(shared)
        codec_q_logits = self.codec_q_head(shared)
        return codec_logits, codec_q_logits


class DANNModel(nn.Module):
    """Complete DANN model for domain-invariant deepfake detection.

    Args:
        backbone: SSL backbone (WavLM, Wav2Vec2, etc.).
        projection: Projection head.
        classifier: Task classifier (bonafide/spoof).
        domain_discriminator: Multi-head domain discriminator.
        pooling: Temporal pooling module.
        lambda_: Initial GRL lambda value.
    """

    def __init__(
        self,
        backbone: nn.Module,
        projection: nn.Module,
        classifier: nn.Module,
        domain_discriminator: nn.Module,
        pooling: nn.Module,
        lambda_: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.classifier = classifier
        self.domain_discriminator = domain_discriminator
        self.pooling = pooling
        self.grl = GradientReversalLayer(lambda_)

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor = None,
        lengths: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: Input audio of shape (B, T).
            attention_mask: Optional attention mask.
            lengths: Optional sequence lengths.

        Returns:
            Dictionary with task_logits, codec_logits, codec_q_logits.
        """
        # Extract features
        features, all_hidden_states = self.backbone(waveform, attention_mask)

        # Project
        projected = self.projection(features)

        # Pool over time
        pooled = self.pooling(projected, lengths)

        # Task classification
        task_logits = self.classifier(pooled)

        # Domain classification (through GRL)
        reversed_features = self.grl(pooled)
        codec_logits, codec_q_logits = self.domain_discriminator(
            reversed_features
        )

        return {
            "task_logits": task_logits,
            "codec_logits": codec_logits,
            "codec_q_logits": codec_q_logits,
            "pooled_features": pooled,
            "all_hidden_states": all_hidden_states,
        }

    def set_lambda(self, lambda_: float) -> None:
        """Update GRL lambda value."""
        self.grl.set_lambda(lambda_)
