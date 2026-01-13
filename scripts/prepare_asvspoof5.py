#!/usr/bin/env python3
"""Prepare ASVspoof 5 manifests from protocol files.

This script parses ASVspoof 5 protocol files (whitespace-separated) and creates
parquet manifests with:
- Absolute audio paths based on ASVSPOOF5_ROOT
- Label encoding: bonafide=0, spoof=1
- Domain normalization: '-' -> 'NONE'
- Persistent vocab files for CODEC and CODEC_Q

Usage:
    export ASVSPOOF5_ROOT=/path/to/asvspoof5
    python scripts/prepare_asvspoof5.py

    # Or with explicit path:
    python scripts/prepare_asvspoof5.py --asvspoof5-root /path/to/asvspoof5
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from asvspoof5_domain_invariant_cm.data.asvspoof5 import (
    KEY_TO_LABEL,
    PROTOCOL_COLUMNS,
    build_vocab,
    load_protocol,
    normalize_domain_value,
    save_vocab,
)
from asvspoof5_domain_invariant_cm.utils.paths import (
    build_audio_path,
    get_asvspoof5_root,
    get_manifests_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ASVspoof 5 manifests from protocol files"
    )
    parser.add_argument(
        "--asvspoof5-root",
        type=Path,
        default=None,
        help="ASVspoof5 data root (overrides ASVSPOOF5_ROOT env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for manifests (default: data/manifests)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that audio files exist",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=100,
        help="Number of audio files to validate (default: 100)",
    )
    return parser.parse_args()


def process_protocol(
    protocol_path: Path,
    asvspoof5_root: Path,
    split_name: str,
) -> pd.DataFrame:
    """Process a protocol file into a manifest DataFrame.

    Args:
        protocol_path: Path to protocol TSV file.
        asvspoof5_root: Root directory of ASVspoof5 data.
        split_name: Name of the split ('train', 'dev', 'eval').

    Returns:
        Processed DataFrame with all required columns.
    """
    logger.info(f"Loading protocol: {protocol_path}")

    df = load_protocol(protocol_path)
    logger.info(f"  Loaded {len(df)} samples")

    # Normalize domain values: '-' -> 'NONE', and '0' -> 'NONE' for codec_q
    df["codec"] = df["codec"].apply(lambda x: normalize_domain_value(x, is_codec_q=False))
    df["codec_q"] = df["codec_q"].apply(lambda x: normalize_domain_value(x, is_codec_q=True))

    # Map KEY to task label: bonafide=0, spoof=1
    df["y_task"] = df["key"].map(KEY_TO_LABEL)

    # Build absolute audio paths based on filename prefix
    def build_path(flac_file: str) -> str:
        return str(build_audio_path(flac_file, asvspoof5_root))

    df["audio_path"] = df["flac_file"].apply(build_path)

    # Add split column
    df["split"] = split_name

    # Log statistics
    logger.info(f"  Keys: {df['key'].value_counts().to_dict()}")
    logger.info(f"  y_task: {df['y_task'].value_counts().to_dict()}")
    logger.info(f"  Unique CODECs: {df['codec'].nunique()}")
    logger.info(f"  Unique CODEC_Qs: {df['codec_q'].nunique()}")

    return df


def encode_domains(
    dfs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Encode domain columns and save vocabularies.

    Args:
        dfs: Dictionary mapping split name to DataFrame.
        output_dir: Directory to save vocabulary files.

    Returns:
        DataFrames with encoded domain columns.
    """
    # Combine all splits to build global vocabulary
    all_codecs = set()
    all_codec_qs = set()

    for df in dfs.values():
        all_codecs.update(df["codec"].unique())
        all_codec_qs.update(df["codec_q"].unique())

    # Build and save vocabularies
    codec_vocab = {v: i for i, v in enumerate(sorted(all_codecs))}
    codec_q_vocab = {v: i for i, v in enumerate(sorted(all_codec_qs))}

    save_vocab(codec_vocab, output_dir / "codec_vocab.json")
    save_vocab(codec_q_vocab, output_dir / "codec_q_vocab.json")

    logger.info(f"Saved codec_vocab.json with {len(codec_vocab)} values")
    logger.info(f"Saved codec_q_vocab.json with {len(codec_q_vocab)} values")

    # Encode domains in each DataFrame
    for split_name, df in dfs.items():
        df["y_codec"] = df["codec"].map(codec_vocab)
        df["y_codec_q"] = df["codec_q"].map(codec_q_vocab)
        dfs[split_name] = df

    return dfs


def validate_audio_paths(df: pd.DataFrame, sample_n: int = 100) -> bool:
    """Validate that audio files exist.

    Args:
        df: Manifest DataFrame with audio_path column.
        sample_n: Number of samples to check.

    Returns:
        True if all sampled paths exist.
    """
    sample = df["audio_path"].sample(min(sample_n, len(df)))
    missing = [p for p in sample if not Path(p).exists()]

    if missing:
        logger.warning(f"Missing {len(missing)}/{len(sample)} sampled files")
        for p in missing[:5]:
            logger.warning(f"  Missing: {p}")
        return False
    else:
        logger.info(f"  Validated {len(sample)} audio paths (all exist)")
        return True


def main():
    args = parse_args()

    # Get ASVspoof5 root
    if args.asvspoof5_root:
        asvspoof5_root = args.asvspoof5_root
    else:
        asvspoof5_root = get_asvspoof5_root()

    logger.info(f"ASVspoof5 root: {asvspoof5_root}")

    # Get output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_manifests_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Protocol file locations
    protocol_dir = asvspoof5_root / "ASVspoof5_protocols"

    splits = {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }

    # Process each split
    dfs = {}
    for split_name, protocol_file in splits.items():
        protocol_path = protocol_dir / protocol_file

        if not protocol_path.exists():
            logger.warning(f"Protocol not found: {protocol_path}")
            continue

        df = process_protocol(protocol_path, asvspoof5_root, split_name)
        dfs[split_name] = df

    if not dfs:
        logger.error("No protocol files found. Check ASVSPOOF5_ROOT.")
        return 1

    # Encode domains and save vocabularies
    dfs = encode_domains(dfs, output_dir)

    # Save manifests
    for split_name, df in dfs.items():
        # Select and order columns
        columns = [
            "audio_path",
            "y_task",
            "y_codec",
            "y_codec_q",
            "speaker_id",
            "flac_file",
            "gender",
            "codec",
            "codec_q",
            "codec_seed",
            "attack_tag",
            "attack_label",
            "key",
            "split",
        ]

        # Only include columns that exist
        columns = [c for c in columns if c in df.columns]
        df = df[columns]

        # Validate audio paths
        if args.validate:
            validate_audio_paths(df, args.sample_n)

        # Save parquet
        output_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved: {output_path} ({len(df)} samples)")

        # Also save CSV for inspection
        csv_path = output_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
