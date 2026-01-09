"""Classification and projection heads."""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """MLP projection head.

    Args:
        input_dim: Input dimension (from backbone).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
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
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features.

        Args:
            x: Input tensor of shape (B, D) or (B, T, D).

        Returns:
            Projected features.
        """
        return self.mlp(x)


class ClassifierHead(nn.Module):
    """Classification head for bonafide/spoof prediction.

    Args:
        input_dim: Input dimension.
        num_classes: Number of output classes.
        hidden_dim: Optional hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
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


class TemporalPooling(nn.Module):
    """Pool temporal dimension of features.

    Args:
        method: Pooling method ('mean', 'attention', 'stats').
        input_dim: Input dimension (for attention pooling).
    """

    def __init__(self, method: str = "mean", input_dim: int = 256):
        super().__init__()
        self.method = method

        if method == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Pool over temporal dimension.

        Args:
            x: Input tensor of shape (B, T, D).
            lengths: Optional sequence lengths of shape (B,).

        Returns:
            Pooled features of shape (B, D).
        """
        if self.method == "mean":
            if lengths is not None:
                # Masked mean pooling
                mask = self._length_to_mask(lengths, x.shape[1])
                x = x * mask.unsqueeze(-1)
                pooled = x.sum(dim=1) / lengths.unsqueeze(-1).float()
            else:
                pooled = x.mean(dim=1)

        elif self.method == "attention":
            # Attention-weighted pooling
            weights = self.attention(x)  # (B, T, 1)
            if lengths is not None:
                mask = self._length_to_mask(lengths, x.shape[1])
                weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            weights = torch.softmax(weights, dim=1)
            pooled = (x * weights).sum(dim=1)

        elif self.method == "stats":
            # Mean + std concatenation
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            pooled = torch.cat([mean, std], dim=-1)

        else:
            raise ValueError(f"Unknown pooling method: {self.method}")

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
