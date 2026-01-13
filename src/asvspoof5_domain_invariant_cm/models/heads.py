"""Classification, projection, and pooling heads."""

from typing import Optional

import torch
import torch.nn as nn


class StatsPooling(nn.Module):
    """Statistics pooling over time: concat(mean, std).

    Output dimension is 2 * input_dim.
    """

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool over temporal dimension using mean + std.

        Args:
            x: Input tensor of shape (B, T, D).
            lengths: Optional sequence lengths of shape (B,).

        Returns:
            Pooled features of shape (B, 2*D).
        """
        if lengths is not None:
            # Masked pooling
            mask = self._length_to_mask(lengths, x.shape[1])
            mask_expanded = mask.unsqueeze(-1).float()

            # Masked mean
            x_masked = x * mask_expanded
            sum_x = x_masked.sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            mean = sum_x / count

            # Masked std
            diff = (x - mean.unsqueeze(1)) * mask_expanded
            var = (diff ** 2).sum(dim=1) / count.clamp(min=1)
            std = torch.sqrt(var + 1e-8)
        else:
            mean = x.mean(dim=1)
            std = x.std(dim=1)

        return torch.cat([mean, std], dim=-1)

    def _length_to_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Convert lengths to boolean mask."""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        return mask


class MeanPooling(nn.Module):
    """Mean pooling over time."""

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool over temporal dimension using mean.

        Args:
            x: Input tensor of shape (B, T, D).
            lengths: Optional sequence lengths of shape (B,).

        Returns:
            Pooled features of shape (B, D).
        """
        if lengths is not None:
            mask = self._length_to_mask(lengths, x.shape[1])
            x = x * mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / lengths.unsqueeze(-1).float().clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return pooled

    def _length_to_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Convert lengths to boolean mask."""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        return mask


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over time.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for attention.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool over temporal dimension using attention weights.

        Args:
            x: Input tensor of shape (B, T, D).
            lengths: Optional sequence lengths of shape (B,).

        Returns:
            Pooled features of shape (B, D).
        """
        weights = self.attention(x)  # (B, T, 1)

        if lengths is not None:
            mask = self._length_to_mask(lengths, x.shape[1])
            weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)

        return pooled

    def _length_to_mask(
        self,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Convert lengths to boolean mask."""
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len, device=lengths.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        return mask


class ProjectionHead(nn.Module):
    """MLP projection head.

    Args:
        input_dim: Input dimension (from pooling).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (repr dimension).
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features.

        Args:
            x: Input tensor of shape (B, D_in).

        Returns:
            Projected features of shape (B, output_dim).
        """
        return self.mlp(x)


class ClassifierHead(nn.Module):
    """Classification head for bonafide/spoof prediction.

    Args:
        input_dim: Input dimension.
        num_classes: Number of output classes.
        hidden_dim: Optional hidden layer dimension (0 = no hidden layer).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify input features.

        Args:
            x: Input tensor of shape (B, D).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.classifier(x)


def create_pooling(method: str = "stats", input_dim: int = 768) -> nn.Module:
    """Factory function to create pooling module.

    Args:
        method: Pooling method ('stats', 'mean', 'attention').
        input_dim: Input dimension (for attention pooling).

    Returns:
        Pooling module.
    """
    if method == "stats":
        return StatsPooling()
    elif method == "mean":
        return MeanPooling()
    elif method == "attention":
        return AttentionPooling(input_dim)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
