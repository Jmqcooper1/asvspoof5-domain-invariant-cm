"""ERM (Empirical Risk Minimization) model without domain adversarial components.

Same pipeline as DANN but without GRL and domain heads.
"""

from typing import Optional

import torch
import torch.nn as nn


class ERMModel(nn.Module):
    """ERM model for deepfake detection (no domain adversarial training).

    Pipeline: backbone -> layer_mix -> stats_pool -> projection -> repr
              task_head(repr) -> task_logits

    Args:
        backbone: SSL backbone (WavLM, Wav2Vec2).
        pooling: Temporal pooling module.
        projection: Projection head (produces repr).
        task_head: Task classifier (bonafide/spoof).
    """

    def __init__(
        self,
        backbone: nn.Module,
        pooling: nn.Module,
        projection: nn.Module,
        task_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.projection = projection
        self.task_head = task_head

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: Input audio of shape (B, T).
            attention_mask: Optional attention mask.
            lengths: Optional sequence lengths.

        Returns:
            Dictionary with:
                - task_logits: (B, 2)
                - repr: (B, repr_dim)
                - all_hidden_states: list of (B, T', D) per layer
        """
        # Backbone: hidden_states -> layer_mix
        mixed, all_hidden_states = self.backbone(waveform, attention_mask)

        # Stats pooling
        pooled = self.pooling(mixed, lengths)  # (B, 2*D) for stats

        # Projection -> repr
        repr_ = self.projection(pooled)  # (B, repr_dim)

        # Task head
        task_logits = self.task_head(repr_)

        return {
            "task_logits": task_logits,
            "repr": repr_,
            "all_hidden_states": all_hidden_states,
        }
