#!/usr/bin/env python3
"""Extract MFCC features for MFCC baseline.

Uses torchaudio MFCC with mean+std pooling over time to produce fixed-size feature vectors.

Usage:
    python scripts/extract_mfcc.py --split train
    python scripts/extract_mfcc.py --split dev
    python scripts/extract_mfcc.py --split eval
    python scripts/extract_mfcc.py --all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.utils.paths import (
    get_features_dir,
    get_manifest_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MFCC features")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default=None,
        help="Split to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all splits",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Number of mel filterbanks",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Max duration in seconds (None = full audio)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/features/mfcc)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for extraction",
    )
    return parser.parse_args()


class MFCCExtractor:
    """MFCC feature extractor with mean+std pooling.

    Args:
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of mel filterbanks.
        sample_rate: Sample rate.
        max_samples: Max samples to use (None = full).
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        n_mels: int = 80,
        sample_rate: int = 16000,
        max_samples: int = None,
    ):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.max_samples = max_samples

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 512,
                "hop_length": 160,  # 10ms hop
                "n_mels": n_mels,
                "f_min": 20,
                "f_max": sample_rate // 2,
            },
        )

    @property
    def feature_dim(self) -> int:
        """Output feature dimension (mean + std)."""
        return self.n_mfcc * 2

    def extract(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract MFCC features with mean+std pooling.

        Args:
            waveform: Input waveform [C, T] or [T].

        Returns:
            Feature vector of shape [n_mfcc * 2].
        """
        # Ensure 2D
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Truncate if needed
        if self.max_samples is not None and waveform.shape[-1] > self.max_samples:
            waveform = waveform[..., :self.max_samples]

        # Extract MFCCs: [C, n_mfcc, T']
        mfcc = self.mfcc_transform(waveform)

        # Take first channel if stereo
        if mfcc.shape[0] > 1:
            mfcc = mfcc[0:1]

        # mfcc shape: [1, n_mfcc, T']
        mfcc = mfcc.squeeze(0)  # [n_mfcc, T']

        # Mean+std pooling over time
        mean = mfcc.mean(dim=-1)  # [n_mfcc]
        std = mfcc.std(dim=-1)   # [n_mfcc]

        # Concatenate
        features = torch.cat([mean, std], dim=0)  # [n_mfcc * 2]

        return features.numpy()


def extract_split(
    split: str,
    extractor: MFCCExtractor,
    output_dir: Path,
) -> None:
    """Extract MFCC features for a split.

    Args:
        split: Split name.
        extractor: MFCC extractor.
        output_dir: Output directory.
    """
    manifest_path = get_manifest_path(split)

    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return

    logger.info(f"Loading manifest: {manifest_path}")
    df = pd.read_parquet(manifest_path)
    logger.info(f"  {len(df)} samples")

    features_list = []
    metadata_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
        audio_path = row["audio_path"]

        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != extractor.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, extractor.sample_rate)
                waveform = resampler(waveform)

            # Extract features
            features = extractor.extract(waveform)
            features_list.append(features)

            # Store metadata
            metadata_list.append({
                "idx": idx,
                "flac_file": row["flac_file"],
                "y_task": row["y_task"],
                "y_codec": row["y_codec"],
                "y_codec_q": row["y_codec_q"],
                "speaker_id": row["speaker_id"],
                "codec": row["codec"],
                "codec_q": row["codec_q"],
            })

        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            continue

    # Stack features
    features_array = np.stack(features_list, axis=0)
    logger.info(f"  Features shape: {features_array.shape}")

    # Save features
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / f"{split}.npy"
    np.save(features_path, features_array)
    logger.info(f"  Saved: {features_path}")

    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = output_dir / f"{split}_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"  Saved: {metadata_path}")


def main():
    args = parse_args()

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_features_dir("mfcc")

    logger.info(f"Output directory: {output_dir}")

    # Max samples
    max_samples = None
    if args.max_duration:
        max_samples = int(args.max_duration * args.sample_rate)

    # Create extractor
    extractor = MFCCExtractor(
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        max_samples=max_samples,
    )

    logger.info(f"MFCC config: n_mfcc={args.n_mfcc}, n_mels={args.n_mels}")
    logger.info(f"Feature dimension: {extractor.feature_dim}")

    # Determine splits to process
    if args.all:
        splits = ["train", "dev", "eval"]
    elif args.split:
        splits = [args.split]
    else:
        logger.error("Specify --split or --all")
        return 1

    # Extract features
    for split in splits:
        extract_split(split, extractor, output_dir)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
