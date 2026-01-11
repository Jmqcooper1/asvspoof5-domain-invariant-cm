"""Model components for domain-invariant deepfake detection."""

from .backbones import (
    LayerWeightedPooling,
    SSLBackbone,
    Wav2Vec2Backbone,
    WavLMBackbone,
    create_backbone,
)
from .dann import (
    DANNModel,
    GradientReversalLayer,
    MultiHeadDomainDiscriminator,
)
from .erm import ERMModel
from .heads import (
    AttentionPooling,
    ClassifierHead,
    MeanPooling,
    ProjectionHead,
    StatsPooling,
    create_pooling,
)
from .losses import DANNLoss, ERMLoss, FocalLoss

__all__ = [
    # Backbones
    "SSLBackbone",
    "WavLMBackbone",
    "Wav2Vec2Backbone",
    "LayerWeightedPooling",
    "create_backbone",
    # Heads
    "ProjectionHead",
    "ClassifierHead",
    "StatsPooling",
    "MeanPooling",
    "AttentionPooling",
    "create_pooling",
    # DANN
    "GradientReversalLayer",
    "MultiHeadDomainDiscriminator",
    "DANNModel",
    # ERM
    "ERMModel",
    # Losses
    "DANNLoss",
    "ERMLoss",
    "FocalLoss",
]
