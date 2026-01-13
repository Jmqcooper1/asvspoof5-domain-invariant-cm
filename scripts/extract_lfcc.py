#!/usr/bin/env python3
"""Extract LFCC (Linear Frequency Cepstral Coefficients) features.

LFCC is a classical feature used in speaker and spoofing detection.
This provides a fallback baseline when TRILLsson is unavailable.

Usage:
    python scripts/extract_lfcc.py --split train
    python scripts/extract_lfcc.py --split dev
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fftpack import dct
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract LFCC features")
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
        help="Output directory",
    )
    parser.add_argument(
        "--n-lfcc",
        type=int,
        default=60,
        help="Number of LFCC coefficients",
    )
    parser.add_argument(
        "--n-filters",
        type=int,
        default=128,
        help="Number of linear filters",
    )
    parser.add_argument(
        "--frame-length-ms",
        type=float,
        default=25.0,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--frame-shift-ms",
        type=float,
        default=10.0,
        help="Frame shift in milliseconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum audio duration in seconds",
    )
    return parser.parse_args()


def linear_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float = None,
) -> np.ndarray:
    """Create linear frequency filterbank.

    Args:
        n_filters: Number of filters.
        n_fft: FFT size.
        sample_rate: Sample rate.
        f_min: Minimum frequency.
        f_max: Maximum frequency.

    Returns:
        Filterbank matrix of shape (n_filters, n_fft // 2 + 1).
    """
    if f_max is None:
        f_max = sample_rate / 2

    # Linear-spaced frequencies
    freqs = np.linspace(f_min, f_max, n_filters + 2)

    # Convert to FFT bins
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    filterbank = np.zeros((n_filters, len(fft_freqs)))

    for i in range(n_filters):
        f_left = freqs[i]
        f_center = freqs[i + 1]
        f_right = freqs[i + 2]

        # Left slope
        left_mask = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        filterbank[i, left_mask] = (fft_freqs[left_mask] - f_left) / (f_center - f_left)

        # Right slope
        right_mask = (fft_freqs >= f_center) & (fft_freqs <= f_right)
        filterbank[i, right_mask] = (f_right - fft_freqs[right_mask]) / (f_right - f_center)

    return filterbank


def extract_lfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_lfcc: int = 60,
    n_filters: int = 128,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """Extract LFCC features from waveform.

    Args:
        waveform: Audio waveform.
        sample_rate: Sample rate.
        n_lfcc: Number of LFCC coefficients.
        n_filters: Number of linear filters.
        frame_length_ms: Frame length in ms.
        frame_shift_ms: Frame shift in ms.

    Returns:
        LFCC features of shape (n_frames, n_lfcc).
    """
    # Frame parameters
    frame_length = int(frame_length_ms * sample_rate / 1000)
    frame_shift = int(frame_shift_ms * sample_rate / 1000)
    n_fft = 2 ** int(np.ceil(np.log2(frame_length)))

    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # Framing
    n_frames = 1 + (len(waveform) - frame_length) // frame_shift
    if n_frames <= 0:
        # Pad if too short
        waveform = np.pad(waveform, (0, frame_length - len(waveform)))
        n_frames = 1

    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * frame_shift
        frames[i] = waveform[start : start + frame_length]

    # Windowing
    window = np.hamming(frame_length)
    frames *= window

    # FFT
    spec = np.abs(np.fft.rfft(frames, n_fft))
    power_spec = spec ** 2

    # Linear filterbank
    filterbank = linear_filterbank(n_filters, n_fft, sample_rate)

    # Apply filterbank
    filtered = np.dot(power_spec, filterbank.T)
    filtered = np.maximum(filtered, 1e-10)
    log_filtered = np.log(filtered)

    # DCT
    lfcc = dct(log_filtered, type=2, axis=1, norm="ortho")[:, :n_lfcc]

    return lfcc


def compute_mean_std(lfcc: np.ndarray) -> np.ndarray:
    """Compute mean and std statistics over frames.

    Args:
        lfcc: LFCC features of shape (n_frames, n_lfcc).

    Returns:
        Statistics vector of shape (2 * n_lfcc,).
    """
    mean = lfcc.mean(axis=0)
    std = lfcc.std(axis=0)
    return np.concatenate([mean, std])


def main():
    args = parse_args()

    from asvspoof5_domain_invariant_cm.utils import get_features_dir, get_manifest_path

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_features_dir("lfcc")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load manifest
    manifest_path = get_manifest_path(args.split)
    logger.info(f"Loading manifest: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    logger.info(f"Loaded {len(df)} samples")

    sample_rate = 16000
    max_samples = int(args.max_duration * sample_rate)

    embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting LFCC"):
        try:
            waveform, sr = sf.read(row["audio_path"], dtype="float32")

            # Resample if needed
            if sr != sample_rate:
                import scipy.signal
                waveform = scipy.signal.resample(
                    waveform, int(len(waveform) * sample_rate / sr)
                )

            # Truncate
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            # Extract LFCC
            lfcc = extract_lfcc(
                waveform,
                sample_rate=sample_rate,
                n_lfcc=args.n_lfcc,
                n_filters=args.n_filters,
                frame_length_ms=args.frame_length_ms,
                frame_shift_ms=args.frame_shift_ms,
            )

            # Compute statistics
            stats = compute_mean_std(lfcc)
            embeddings.append(stats)

        except Exception as e:
            logger.warning(f"Error processing {row['audio_path']}: {e}")
            # Use zeros for failed files
            embeddings.append(np.zeros(2 * args.n_lfcc, dtype=np.float32))

    embeddings = np.stack(embeddings, axis=0)
    logger.info(f"Extracted features shape: {embeddings.shape}")

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
