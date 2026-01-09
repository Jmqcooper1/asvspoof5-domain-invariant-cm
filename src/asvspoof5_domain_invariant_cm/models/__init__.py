"""Model components for domain-invariant deepfake detection."""

from .backbones import SSLBackbone, WavLMBackbone, Wav2Vec2Backbone
from .dann import GradientReversalLayer, MultiHeadDomainDiscriminator
from .heads import ClassifierHead, ProjectionHead
from .losses import DANNLoss

__all__ = [
    "SSLBackbone",
    "WavLMBackbone",
    "Wav2Vec2Backbone",
    "GradientReversalLayer",
    "MultiHeadDomainDiscriminator",
    "ClassifierHead",
    "ProjectionHead",
    "DANNLoss",
]
