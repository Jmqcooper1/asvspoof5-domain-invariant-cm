#!/usr/bin/env python3
"""Parse ASVspoof 5 protocol files and create unified manifests.

ASVspoof 5 protocol files are whitespace-separated (not tab-separated)
despite the .tsv extension. This script handles that correctly.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Column names for Track 1 protocol files
PROTOCOL_COLUMNS = [
    "speaker_id",
    "flac_file",
    "gender",
    "codec",
    "codec_q",
    "codec_seed",
    "attack_tag",
    "attack_label",
    "key",
    "tmp",
]


def load_protocol(path: Path) -> pd.DataFrame:
    """Load ASVspoof 5 protocol file (whitespace-separated).

    Args:
        path: Path to protocol TSV file.

    Returns:
        DataFrame with protocol data.
    """
    logger.info(f"Loading protocol: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",  # Whitespace-separated, NOT tab
        header=None,
        names=PROTOCOL_COLUMNS,
        dtype=str,
    )

    logger.info(f"  Loaded {len(df)} samples")
    logger.info(f"  Keys: {df['key'].value_counts().to_dict()}")

    return df


def add_audio_paths(
    df: pd.DataFrame,
    audio_dir: Path,
) -> pd.DataFrame:
    """Add full audio paths to manifest.

    Args:
        df: Protocol DataFrame.
        audio_dir: Directory containing audio files.

    Returns:
        DataFrame with audio_path column added.
    """
    df = df.copy()
    df["audio_path"] = df["flac_file"].apply(lambda x: str(audio_dir / x))
    return df


def validate_audio_paths(df: pd.DataFrame, sample_n: int = 100) -> None:
    """Validate that audio files exist.

    Args:
        df: Manifest DataFrame with audio_path column.
        sample_n: Number of samples to check.
    """
    sample = df["audio_path"].sample(min(sample_n, len(df)))
    missing = [p for p in sample if not Path(p).exists()]

    if missing:
        logger.warning(f"Missing {len(missing)}/{len(sample)} sampled files")
        logger.warning(f"  Example: {missing[0]}")
    else:
        logger.info(f"  Validated {len(sample)} audio paths (all exist)")


def compute_domain_stats(df: pd.DataFrame) -> None:
    """Log domain statistics for DANN setup."""
    logger.info("Domain statistics:")

    for col in ["codec", "codec_q"]:
        n_unique = df[col].nunique()
        logger.info(f"  {col}: {n_unique} unique values")


def main():
    parser = argparse.ArgumentParser(description="Create ASVspoof 5 manifests")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/asvspoof5"),
        help="Root directory of ASVspoof 5 data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for manifests",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that audio files exist",
    )
    args = parser.parse_args()

    # Paths
    protocol_dir = args.data_root / "ASVspoof5_protocols"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = {
        "train": {
            "protocol": "ASVspoof5.train.tsv",
            "audio_dir": "flac_T",
        },
        "dev": {
            "protocol": "ASVspoof5.dev.track_1.tsv",
            "audio_dir": "flac_D",
        },
        "eval": {
            "protocol": "ASVspoof5.eval.track_1.tsv",
            "audio_dir": "flac_E_eval",
        },
    }

    for split_name, split_info in splits.items():
        protocol_path = protocol_dir / split_info["protocol"]
        audio_dir = args.data_root / split_info["audio_dir"]

        if not protocol_path.exists():
            logger.warning(f"Protocol not found: {protocol_path}")
            continue

        # Load and process
        df = load_protocol(protocol_path)
        df = add_audio_paths(df, audio_dir)

        # Add split column
        df["split"] = split_name

        # Validate
        if args.validate and audio_dir.exists():
            validate_audio_paths(df)

        # Domain stats (for DANN)
        compute_domain_stats(df)

        # Save
        output_path = args.output_dir / f"{split_name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved: {output_path}")

        # Also save CSV for inspection
        csv_path = args.output_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()
