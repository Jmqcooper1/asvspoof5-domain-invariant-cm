"""SSL backbone wrappers with layer output extraction."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model, WavLMModel


class LayerWeightedPooling(nn.Module):
    """Learn weighted combination of layer outputs.

    Args:
        num_layers: Number of transformer layers.
        init_lower_bias: If True, initialize with higher weights for lower layers.
    """

    def __init__(self, num_layers: int, init_lower_bias: bool = True):
        super().__init__()
        if init_lower_bias:
            # Higher weights for lower layers
            weights = torch.linspace(1.0, 0.1, num_layers)
        else:
            weights = torch.ones(num_layers)
        self.weights = nn.Parameter(weights)

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of layer outputs.

        Args:
            hidden_states: List of tensors of shape (B, T, D).

        Returns:
            Weighted output of shape (B, T, D).
        """
        weights = F.softmax(self.weights, dim=0)
        stacked = torch.stack(hidden_states, dim=0)  # (L, B, T, D)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted


class SSLBackbone(ABC, nn.Module):
    """Abstract base class for SSL backbones."""

    @abstractmethod
    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Extract features from waveform.

        Args:
            waveform: Input audio of shape (B, T).
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (final_output, all_hidden_states).
        """
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Hidden dimension of the backbone."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers."""
        pass


class WavLMBackbone(SSLBackbone):
    """WavLM backbone wrapper.

    Args:
        pretrained: HuggingFace model name or path.
        freeze: If True, freeze backbone parameters.
        layer_selection: Method for selecting layers ('all', 'first_k', 'last_k').
        k: Number of layers for first_k/last_k selection.
    """

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        layer_selection: str = "all",
        k: int = 6,
        init_lower_bias: bool = True,
    ):
        super().__init__()
        self.model = WavLMModel.from_pretrained(
            pretrained,
            output_hidden_states=True,
        )
        self._hidden_size = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.layer_selection = layer_selection
        self.k = k

        # Layer weighting
        if layer_selection == "all":
            num_weighted_layers = self._num_layers
        elif layer_selection in ("first_k", "last_k"):
            num_weighted_layers = k
        else:
            num_weighted_layers = self._num_layers

        self.layer_pooling = LayerWeightedPooling(
            num_weighted_layers,
            init_lower_bias=init_lower_bias,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = self.model(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states (skip input embedding, index 0)
        hidden_states = outputs.hidden_states[1:]  # (L, B, T, D)

        # Select layers
        if self.layer_selection == "first_k":
            selected = list(hidden_states[: self.k])
        elif self.layer_selection == "last_k":
            selected = list(hidden_states[-self.k :])
        else:
            selected = list(hidden_states)

        # Weighted pooling over layers
        pooled = self.layer_pooling(selected)

        return pooled, list(hidden_states)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers


class Wav2Vec2Backbone(SSLBackbone):
    """Wav2Vec 2.0 backbone wrapper.

    Args:
        pretrained: HuggingFace model name or path.
        freeze: If True, freeze backbone parameters.
        layer_selection: Method for selecting layers.
        k: Number of layers for first_k/last_k selection.
    """

    def __init__(
        self,
        pretrained: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        layer_selection: str = "all",
        k: int = 6,
        init_lower_bias: bool = True,
    ):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(
            pretrained,
            output_hidden_states=True,
        )
        self._hidden_size = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.layer_selection = layer_selection
        self.k = k

        # Layer weighting
        if layer_selection == "all":
            num_weighted_layers = self._num_layers
        elif layer_selection in ("first_k", "last_k"):
            num_weighted_layers = k
        else:
            num_weighted_layers = self._num_layers

        self.layer_pooling = LayerWeightedPooling(
            num_weighted_layers,
            init_lower_bias=init_lower_bias,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = self.model(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states (skip input embedding)
        hidden_states = outputs.hidden_states[1:]

        # Select layers
        if self.layer_selection == "first_k":
            selected = list(hidden_states[: self.k])
        elif self.layer_selection == "last_k":
            selected = list(hidden_states[-self.k :])
        else:
            selected = list(hidden_states)

        # Weighted pooling over layers
        pooled = self.layer_pooling(selected)

        return pooled, list(hidden_states)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers
