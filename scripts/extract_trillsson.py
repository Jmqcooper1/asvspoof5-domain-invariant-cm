#!/usr/bin/env python3
"""Extract TRILLsson embeddings from audio files.

TRILLsson is a non-semantic audio embedding model from Google.
This script extracts embeddings and saves them for downstream classification.

Requirements:
    uv sync --extra trillsson

Offline extraction workflow (recommended for Snellius):
1. On local machine with TensorFlow:
   pip install tensorflow tensorflow-hub torchaudio numpy pandas
   python scripts/extract_trillsson.py --split train --output-dir data/features/trillsson
   python scripts/extract_trillsson.py --split dev --output-dir data/features/trillsson
2. Upload to Snellius:
   rsync -avz data/features/trillsson/ snellius:$ASVSPOOF5_ROOT/../features/trillsson/
3. On Snellius, run classifier training (no TF needed):
   python scripts/train_trillsson.py --config configs/trillsson_baseline.yaml

Usage:
    python scripts/extract_trillsson.py --split train
    python scripts/extract_trillsson.py --split dev
    python scripts/extract_trillsson.py --split eval
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract TRILLsson embeddings")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        required=True,
        help="Data split to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/features/trillsson)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trillsson3",
        choices=["trillsson1", "trillsson2", "trillsson3", "trillsson4", "trillsson5"],
        help="TRILLsson model variant",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum audio duration in seconds",
    )
    return parser.parse_args()


def load_trillsson_model(model_name: str):
    """Load TRILLsson model from TensorFlow Hub.

    Args:
        model_name: Model variant name.

    Returns:
        TensorFlow Hub model.
    """
    try:
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError(
            "TensorFlow Hub is required for TRILLsson extraction.\n"
            "Install with: uv sync --extra trillsson"
        )

    # TRILLsson model URLs
    model_urls = {
        "trillsson1": "https://tfhub.dev/google/trillsson1/1",
        "trillsson2": "https://tfhub.dev/google/trillsson2/1",
        "trillsson3": "https://tfhub.dev/google/trillsson3/1",
        "trillsson4": "https://tfhub.dev/google/trillsson4/1",
        "trillsson5": "https://tfhub.dev/google/trillsson5/1",
    }

    if model_name not in model_urls:
        raise ValueError(f"Unknown model: {model_name}")

    logger.info(f"Loading {model_name} from TensorFlow Hub...")
    model = hub.load(model_urls[model_name])

    return model


def load_audio_for_trillsson(audio_path: str, sample_rate: int = 16000, max_samples: int = None):
    """Load audio file for TRILLsson.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate.
        max_samples: Maximum samples to load.

    Returns:
        Audio waveform as numpy array.
    """
    try:
        import torchaudio
    except ImportError:
        import soundfile as sf
        waveform, sr = sf.read(audio_path, dtype="float32")
        if sr != sample_rate:
            # Simple resampling (not ideal but works)
            import scipy.signal
            waveform = scipy.signal.resample(
                waveform, int(len(waveform) * sample_rate / sr)
            )
        if max_samples and len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        return waveform

    waveform, sr = torchaudio.load(audio_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.squeeze().numpy()

    if max_samples and len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return waveform


def extract_embeddings(
    model,
    audio_paths: list[str],
    sample_rate: int = 16000,
    max_duration: float = 6.0,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract embeddings from audio files.

    Args:
        model: TRILLsson model.
        audio_paths: List of audio file paths.
        sample_rate: Sample rate.
        max_duration: Maximum duration in seconds.
        batch_size: Batch size.

    Returns:
        Embeddings array of shape (N, D).
    """
    import tensorflow as tf

    max_samples = int(max_duration * sample_rate)
    all_embeddings = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting"):
        batch_paths = audio_paths[i : i + batch_size]

        # Load audio files
        batch_audio = []
        for path in batch_paths:
            try:
                audio = load_audio_for_trillsson(path, sample_rate, max_samples)
                # Pad if too short
                if len(audio) < max_samples:
                    audio = np.pad(audio, (0, max_samples - len(audio)))
                batch_audio.append(audio)
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                # Use zeros for failed files
                batch_audio.append(np.zeros(max_samples, dtype=np.float32))

        batch_audio = np.stack(batch_audio, axis=0)

        # Convert to TensorFlow tensor
        batch_tensor = tf.constant(batch_audio, dtype=tf.float32)

        # Extract embeddings
        embeddings = model(batch_tensor)["embedding"]
        embeddings = embeddings.numpy()

        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def main():
    args = parse_args()

    # Import here to check TensorFlow availability early
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")

        # Disable GPU if needed
        # tf.config.set_visible_devices([], 'GPU')
    except ImportError:
        logger.error(
            "TensorFlow is required for TRILLsson extraction.\n"
            "Install with: uv sync --extra trillsson"
        )
        return 1

    # Get paths
    from asvspoof5_domain_invariant_cm.utils import get_features_dir, get_manifest_path

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_features_dir("trillsson")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load manifest
    manifest_path = get_manifest_path(args.split)
    logger.info(f"Loading manifest: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    logger.info(f"Loaded {len(df)} samples")

    # Get audio paths
    audio_paths = df["audio_path"].tolist()

    # Load model
    model = load_trillsson_model(args.model)

    # Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings = extract_embeddings(
        model,
        audio_paths,
        sample_rate=16000,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
    )

    logger.info(f"Extracted embeddings shape: {embeddings.shape}")

    # Save embeddings
    embeddings_path = output_dir / f"{args.split}.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings: {embeddings_path}")

    # Save metadata
    metadata = df[["flac_file", "y_task", "y_codec", "y_codec_q", "codec", "codec_q"]].copy()
    metadata["embedding_idx"] = range(len(metadata))

    metadata_path = output_dir / f"{args.split}_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata: {metadata_path}")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
