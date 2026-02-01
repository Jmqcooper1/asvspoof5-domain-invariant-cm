"""Synthetic codec augmentation for domain-adversarial training.

Since ASVspoof5 train/dev sets have no codec diversity (all samples are uncoded),
we need to synthetically create codec domains via augmentation to make DANN meaningful.

This module provides:
- Codec augmentation using ffmpeg subprocess
- Mapping from synthetic domains to approximate ASVspoof5 codec families
- Caching support for offline augmentation

Target domains (aligned with ASVspoof5 codec families):
- NONE: clean/original (no augmentation)
- MP3: maps to C05 (mp3_wb), C07 (mp3_encodec_wb)
- AAC: maps to C06 (m4a_wb)
- OPUS: maps to C01 (opus_wb), C08 (opus_nb)
- SPEEX: maps to C03 (speex_wb), C10 (speex_nb) - requires ffmpeg with speex support
- AMR: maps to C02 (amr_wb), C09 (amr_nb) - requires ffmpeg with amr support
"""

import hashlib
import logging
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio using soundfile (avoids torchcodec issues)."""
    data, sr = sf.read(path, dtype="float32")
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T  # [T, C] -> [C, T]
    return waveform, sr


def _save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save audio using soundfile (avoids torchcodec issues)."""
    if waveform.ndim == 2:
        # [C, T] -> [T, C] for soundfile
        data = waveform.T.numpy()
    else:
        data = waveform.numpy()
    sf.write(path, data, sample_rate)

# Domain vocabulary for synthetic codecs
SYNTHETIC_CODEC_VOCAB = {
    "NONE": 0,
    "MP3": 1,
    "AAC": 2,
    "OPUS": 3,
    "SPEEX": 4,
    "AMR": 5,
}

# Quality levels (1=lowest, 5=highest bitrate)
SYNTHETIC_QUALITY_VOCAB = {
    "NONE": 0,  # Quality undefined for uncoded
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}

# Bitrate configurations per codec and quality level (kbps)
CODEC_BITRATES = {
    "MP3": {
        1: 64,
        2: 96,
        3: 128,
        4: 192,
        5: 256,
    },
    "AAC": {
        1: 32,
        2: 64,
        3: 96,
        4: 128,
        5: 192,
    },
    "OPUS": {
        1: 12,
        2: 24,
        3: 48,
        4: 64,
        5: 96,
    },
    "SPEEX": {
        1: 8,
        2: 16,
        3: 24,
        4: 32,
        5: 44,
    },
    "AMR": {
        1: 6,
        2: 9,
        3: 12,
        4: 18,
        5: 23,
    },
}


@dataclass
class CodecAugmentConfig:
    """Configuration for codec augmentation.

    Attributes:
        enabled: Whether augmentation is enabled.
        codec_prob: Probability of applying any codec (vs keeping clean).
        codecs: List of codec names to use.
        qualities: List of quality levels to use.
        cache_dir: Directory for caching augmented audio (None = on-the-fly).
        sample_rate: Target sample rate.
    """

    enabled: bool = True
    codec_prob: float = 0.5
    codecs: list[str] = field(default_factory=lambda: ["MP3", "AAC", "OPUS"])
    qualities: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    cache_dir: Optional[Path] = None
    sample_rate: int = 16000


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_codec_support(codec: str) -> bool:
    """Check if ffmpeg supports a codec.

    Args:
        codec: Codec name (MP3, AAC, OPUS, SPEEX, AMR).

    Returns:
        True if codec is supported.
    """
    encoder_map = {
        "MP3": "libmp3lame",
        "AAC": "aac",
        "OPUS": "libopus",
        "SPEEX": "libspeex",
        "AMR": "libopencore_amrnb",
    }

    encoder = encoder_map.get(codec)
    if not encoder:
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
        )
        return encoder in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_cache_key(audio_path: str, codec: str, quality: int) -> str:
    """Generate cache key for augmented audio.

    Args:
        audio_path: Original audio path.
        codec: Codec name.
        quality: Quality level.

    Returns:
        Cache key string.
    """
    path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:12]
    return f"{path_hash}_{codec}_{quality}"


def apply_codec_ffmpeg(
    input_path: Path,
    output_path: Path,
    codec: str,
    bitrate_kbps: int,
    sample_rate: int = 16000,
) -> bool:
    """Apply codec compression using ffmpeg.

    Args:
        input_path: Input audio file path.
        output_path: Output audio file path.
        codec: Codec name (MP3, AAC, OPUS, SPEEX, AMR).
        bitrate_kbps: Target bitrate in kbps.
        sample_rate: Target sample rate.

    Returns:
        True if successful.
    """
    # Map codec to ffmpeg encoder and format
    codec_config = {
        "MP3": {
            "encoder": "libmp3lame",
            "format": "mp3",
            "ext": ".mp3",
        },
        "AAC": {
            "encoder": "aac",
            "format": "adts",
            "ext": ".aac",
        },
        "OPUS": {
            "encoder": "libopus",
            "format": "opus",
            "ext": ".opus",
        },
        "SPEEX": {
            "encoder": "libspeex",
            "format": "ogg",
            "ext": ".spx",
        },
        "AMR": {
            "encoder": "libopencore_amrnb",
            "format": "amr",
            "ext": ".amr",
            "sample_rate": 8000,  # AMR-NB requires 8kHz
        },
    }

    if codec not in codec_config:
        logger.warning(f"Unknown codec: {codec}")
        return False

    config = codec_config[codec]
    target_sr = config.get("sample_rate", sample_rate)

    # Initialize to None to avoid UnboundLocalError in finally
    tmp_encoded = None

    try:
        with tempfile.NamedTemporaryFile(suffix=config["ext"], delete=False) as tmp:
            tmp_encoded = Path(tmp.name)

        # Encode
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-ar", str(target_sr),
            "-ac", "1",  # mono
            "-c:a", config["encoder"],
            "-b:a", f"{bitrate_kbps}k",
            "-f", config["format"],
            str(tmp_encoded),
        ]

        subprocess.run(
            encode_cmd,
            capture_output=True,
            check=True,
        )

        # Decode back to wav/flac
        decode_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(tmp_encoded),
            "-ar", str(sample_rate),  # resample back to original rate
            "-ac", "1",
            str(output_path),
        ]

        subprocess.run(
            decode_cmd,
            capture_output=True,
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        logger.warning(f"FFmpeg error for {codec}: {e.stderr.decode()[:200]}")
        return False
    finally:
        if tmp_encoded is not None and tmp_encoded.exists():
            tmp_encoded.unlink()


class CodecAugmentor:
    """Codec augmentor with optional caching.

    Args:
        config: Augmentation configuration.
    """

    def __init__(self, config: CodecAugmentConfig):
        self.config = config
        self._ffmpeg_available = None
        self._supported_codecs = None
        self._codec_vocab = None

        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    @property
    def ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available (cached)."""
        if self._ffmpeg_available is None:
            self._ffmpeg_available = check_ffmpeg_available()
        return self._ffmpeg_available

    @property
    def supported_codecs(self) -> list[str]:
        """Get list of supported codecs (cached)."""
        if self._supported_codecs is None:
            self._supported_codecs = [
                c for c in self.config.codecs if check_codec_support(c)
            ]
            # Log informative warnings about codec support
            if len(self._supported_codecs) < 2:
                logger.error(
                    "CRITICAL: Only %d codec(s) supported by ffmpeg! "
                    "DANN requires domain diversity for meaningful training. "
                    "Requested codecs: %s, Supported: %s. "
                    "Check your ffmpeg installation:\n"
                    "  ffmpeg -encoders | grep -E 'mp3|aac|opus|speex|amr'\n"
                    "Consider installing ffmpeg with more codec support.",
                    len(self._supported_codecs),
                    self.config.codecs,
                    self._supported_codecs or "(none)",
                )
            elif len(self._supported_codecs) < len(self.config.codecs):
                unsupported = set(self.config.codecs) - set(self._supported_codecs)
                logger.warning(
                    "Some requested codecs not supported by ffmpeg: %s. "
                    "Using: %s",
                    unsupported,
                    self._supported_codecs,
                )
        return self._supported_codecs

    @property
    def codec_vocab(self) -> dict[str, int]:
        """Dynamic synthetic codec vocab aligned to actual supported codecs.

        Always includes:
        - NONE: 0

        Then assigns consecutive IDs to the codecs that are both requested and
        supported by ffmpeg, in the order they appear in the config.

        This guarantees:
        - discriminator output dim == number of actually reachable codec classes
        - y_codec_aug indices are consistent with saved vocab JSON
        """
        if self._codec_vocab is None:
            supported = self.supported_codecs
            self._codec_vocab = {"NONE": 0}
            for i, codec in enumerate(supported, start=1):
                self._codec_vocab[codec] = i
        return self._codec_vocab

    def augment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        audio_path: Optional[str] = None,
        force: bool = False,
    ) -> tuple[torch.Tensor, str, int]:
        """Apply random codec augmentation.

        Args:
            waveform: Input waveform tensor [C, T] or [T].
            sample_rate: Sample rate.
            audio_path: Original audio path (for caching).
            force: If True, skip the codec_prob check and always apply
                augmentation. Used by ASVspoof5Dataset on cache miss when
                the caller has already rolled codec_prob.

        Returns:
            Tuple of (augmented_waveform, codec_name, quality_level).
            codec_name is "NONE" and quality is 0 if no augmentation applied.
        """
        if not self.config.enabled:
            return waveform, "NONE", 0

        # Decide whether to augment (skip check when caller already decided)
        if not force and random.random() > self.config.codec_prob:
            return waveform, "NONE", 0

        if not self.ffmpeg_available or not self.supported_codecs:
            return waveform, "NONE", 0

        # Sample codec and quality
        codec = random.choice(self.supported_codecs)
        quality = random.choice(self.config.qualities)

        # Check cache
        if self.cache_dir and audio_path:
            cache_key = get_cache_key(audio_path, codec, quality)
            cache_path = self.cache_dir / f"{cache_key}.flac"

            if cache_path.exists():
                try:
                    cached_wav, _ = _load_audio(str(cache_path))
                    return cached_wav, codec, quality
                except Exception as e:
                    # Cache file is corrupted, delete it and re-augment
                    logger.debug(f"Corrupted cache file {cache_path}, removing: {e}")
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass

        # Apply augmentation
        augmented = self._apply_codec(waveform, sample_rate, codec, quality)

        # Save to cache (atomic write to avoid corruption)
        if self.cache_dir and audio_path and augmented is not None:
            cache_key = get_cache_key(audio_path, codec, quality)
            cache_path = self.cache_dir / f"{cache_key}.flac"
            temp_path = cache_path.with_suffix(".tmp")
            try:
                _save_audio(str(temp_path), augmented, sample_rate)
                os.replace(str(temp_path), str(cache_path))  # Atomic on POSIX
            except Exception as e:
                logger.debug(f"Failed to cache augmented audio: {e}")
                if temp_path.exists():
                    temp_path.unlink()

        if augmented is not None:
            return augmented, codec, quality
        else:
            return waveform, "NONE", 0

    def _apply_codec(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        codec: str,
        quality: int,
    ) -> Optional[torch.Tensor]:
        """Apply codec compression.

        Args:
            waveform: Input waveform.
            sample_rate: Sample rate.
            codec: Codec name.
            quality: Quality level (1-5).

        Returns:
            Augmented waveform or None if failed.
        """
        bitrate = CODEC_BITRATES.get(codec, {}).get(quality)
        if bitrate is None:
            return None

        # Initialize to None to avoid UnboundLocalError in finally
        input_path = None
        output_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                input_path = Path(tmp_in.name)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = Path(tmp_out.name)

            # Ensure 2D tensor for saving
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # Save input
            _save_audio(str(input_path), waveform, sample_rate)

            # Apply codec
            success = apply_codec_ffmpeg(
                input_path,
                output_path,
                codec,
                bitrate,
                sample_rate,
            )

            if success and output_path.exists():
                augmented, _ = _load_audio(str(output_path))
                return augmented
            else:
                return None

        except Exception as e:
            logger.debug(f"Codec augmentation failed: {e}")
            return None

        finally:
            if input_path is not None and input_path.exists():
                input_path.unlink()
            if output_path is not None and output_path.exists():
                output_path.unlink()

    def get_domain_labels(
        self,
        codec: str,
        quality: int,
    ) -> tuple[int, int]:
        """Get domain label IDs for codec and quality.

        Args:
            codec: Codec name.
            quality: Quality level (0-5, 0=NONE).

        Returns:
            Tuple of (codec_id, quality_id).
        """
        codec_id = self.codec_vocab.get(codec, 0)
        quality_id = SYNTHETIC_QUALITY_VOCAB.get(str(quality), 0)
        return codec_id, quality_id


def create_augmentor(config: dict) -> Optional[CodecAugmentor]:
    """Create codec augmentor from config dict.

    Args:
        config: Augmentation config dict with keys:
            - enabled: bool
            - codec_prob: float
            - codecs: list[str]
            - qualities: list[int]
            - cache_dir: str (optional)

    Returns:
        CodecAugmentor instance or None if disabled.
    """
    if not config.get("enabled", False):
        return None

    aug_config = CodecAugmentConfig(
        enabled=config.get("enabled", True),
        codec_prob=config.get("codec_prob", 0.5),
        codecs=config.get("codecs", ["MP3", "AAC", "OPUS"]),
        qualities=config.get("qualities", [1, 2, 3, 4, 5]),
        cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") else None,
        sample_rate=config.get("sample_rate", 16000),
    )

    return CodecAugmentor(aug_config)
