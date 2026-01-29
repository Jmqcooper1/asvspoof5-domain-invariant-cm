"""SSL backbone wrappers with layer output extraction."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, WavLMModel


class LayerWeightedPooling(nn.Module):
    """Learn weighted combination of layer outputs.

    Args:
        num_layers: Number of layers to weight.
        init_lower_bias: If True, initialize with higher weights for lower layers.
    """

    def __init__(self, num_layers: int, init_lower_bias: bool = True):
        super().__init__()
        if init_lower_bias:
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
            Tuple of (layer_mixed_output [B, T', D], all_hidden_states).
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


def select_layers(
    hidden_states: tuple[torch.Tensor, ...],
    method: str,
    k: int = 6,
    layer_indices: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """Select layers according to method.

    Args:
        hidden_states: Tuple of hidden states (includes embedding layer at index 0).
        method: Selection method ('weighted', 'first_k', 'last_k', 'specific').
        k: Number of layers for first_k/last_k.
        layer_indices: Explicit layer indices for 'specific' method.

    Returns:
        List of selected hidden states (excludes embedding layer).
    """
    # Skip embedding layer (index 0)
    transformer_states = list(hidden_states[1:])

    if method == "first_k":
        return transformer_states[:k]
    elif method == "last_k":
        return transformer_states[-k:]
    elif method == "specific" and layer_indices is not None:
        return [transformer_states[i] for i in layer_indices if i < len(transformer_states)]
    else:  # weighted or all
        return transformer_states


class WavLMBackbone(SSLBackbone):
    """WavLM backbone wrapper.

    Args:
        pretrained: HuggingFace model name or path.
        freeze: If True, freeze backbone parameters.
        layer_selection: Method for selecting layers ('weighted', 'first_k', 'last_k', 'specific').
        k: Number of layers for first_k/last_k selection.
        layer_indices: Explicit layer indices for 'specific' method.
        init_lower_bias: If True, initialize layer weights with higher values for lower layers.
    """

    def __init__(
        self,
        pretrained: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        layer_selection: str = "weighted",
        k: int = 6,
        layer_indices: Optional[list[int]] = None,
        init_lower_bias: bool = True,
    ):
        super().__init__()
        self._freeze = freeze
        self.model = WavLMModel.from_pretrained(
            pretrained,
            output_hidden_states=True,
        )
        self._hidden_size = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # Important: even if frozen, nn.Module.train() would re-enable dropout.
            # Keep the HF backbone in eval mode during training for numerical stability.
            self.model.eval()

        self.layer_selection = layer_selection
        self.k = k
        self.layer_indices = layer_indices

        # Determine number of layers for weighting
        if layer_selection == "first_k" or layer_selection == "last_k":
            num_weighted = k
        elif layer_selection == "specific" and layer_indices:
            num_weighted = len(layer_indices)
        else:
            num_weighted = self._num_layers

        self.layer_pooling = LayerWeightedPooling(
            num_weighted,
            init_lower_bias=init_lower_bias,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            self.model.eval()
        return self

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

        all_hidden_states = list(outputs.hidden_states[1:])  # skip embedding

        selected = select_layers(
            outputs.hidden_states,
            self.layer_selection,
            self.k,
            self.layer_indices,
        )

        mixed = self.layer_pooling(selected)

        return mixed, all_hidden_states

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
        layer_indices: Explicit layer indices for 'specific' method.
        init_lower_bias: If True, initialize layer weights with higher values for lower layers.
    """

    def __init__(
        self,
        pretrained: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        layer_selection: str = "weighted",
        k: int = 6,
        layer_indices: Optional[list[int]] = None,
        init_lower_bias: bool = True,
    ):
        super().__init__()
        self._freeze = freeze
        self.model = Wav2Vec2Model.from_pretrained(
            pretrained,
            output_hidden_states=True,
        )
        self._hidden_size = self.model.config.hidden_size
        self._num_layers = self.model.config.num_hidden_layers

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # Important: even if frozen, nn.Module.train() would re-enable dropout.
            # Keep the HF backbone in eval mode during training for numerical stability.
            self.model.eval()

        self.layer_selection = layer_selection
        self.k = k
        self.layer_indices = layer_indices

        # Determine number of layers for weighting
        if layer_selection == "first_k" or layer_selection == "last_k":
            num_weighted = k
        elif layer_selection == "specific" and layer_indices:
            num_weighted = len(layer_indices)
        else:
            num_weighted = self._num_layers

        self.layer_pooling = LayerWeightedPooling(
            num_weighted,
            init_lower_bias=init_lower_bias,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze:
            self.model.eval()
        return self

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

        all_hidden_states = list(outputs.hidden_states[1:])  # skip embedding

        selected = select_layers(
            outputs.hidden_states,
            self.layer_selection,
            self.k,
            self.layer_indices,
        )

        mixed = self.layer_pooling(selected)

        return mixed, all_hidden_states

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers


def create_backbone(
    name: str,
    pretrained: str,
    freeze: bool = True,
    layer_selection: str = "weighted",
    k: int = 6,
    layer_indices: Optional[list[int]] = None,
    init_lower_bias: bool = True,
) -> SSLBackbone:
    """Factory function to create SSL backbone.

    Args:
        name: Backbone name ('wavlm_base_plus', 'wav2vec2_base').
        pretrained: HuggingFace model name or path.
        freeze: Whether to freeze backbone.
        layer_selection: Layer selection method.
        k: Number of layers for first_k/last_k.
        layer_indices: Explicit layer indices for 'specific'.
        init_lower_bias: Initialize with lower layer bias.

    Returns:
        SSL backbone instance.
    """
    if "wavlm" in name.lower():
        return WavLMBackbone(
            pretrained=pretrained,
            freeze=freeze,
            layer_selection=layer_selection,
            k=k,
            layer_indices=layer_indices,
            init_lower_bias=init_lower_bias,
        )
    elif "wav2vec" in name.lower():
        return Wav2Vec2Backbone(
            pretrained=pretrained,
            freeze=freeze,
            layer_selection=layer_selection,
            k=k,
            layer_indices=layer_indices,
            init_lower_bias=init_lower_bias,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
