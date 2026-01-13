"""Data loading and processing for ASVspoof 5."""

from .asvspoof5 import (
    KEY_TO_LABEL,
    PROTOCOL_COLUMNS,
    ASVspoof5Dataset,
    build_vocab,
    create_dataloader,
    get_codec_seed_groups,
    load_protocol,
    load_vocab,
    normalize_domain_value,
    save_vocab,
)
from .audio import AudioCollator, collate_audio_batch, crop_or_pad, load_waveform
from .codec_augment import (
    SYNTHETIC_CODEC_VOCAB,
    SYNTHETIC_QUALITY_VOCAB,
    CodecAugmentConfig,
    CodecAugmentor,
    create_augmentor,
)

__all__ = [
    "PROTOCOL_COLUMNS",
    "KEY_TO_LABEL",
    "load_protocol",
    "get_codec_seed_groups",
    "normalize_domain_value",
    "build_vocab",
    "save_vocab",
    "load_vocab",
    "ASVspoof5Dataset",
    "create_dataloader",
    "load_waveform",
    "crop_or_pad",
    "collate_audio_batch",
    "AudioCollator",
    # Codec augmentation
    "SYNTHETIC_CODEC_VOCAB",
    "SYNTHETIC_QUALITY_VOCAB",
    "CodecAugmentConfig",
    "CodecAugmentor",
    "create_augmentor",
]
