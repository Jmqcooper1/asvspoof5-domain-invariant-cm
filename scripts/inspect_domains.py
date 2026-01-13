#!/usr/bin/env python3
"""Inspect domain labels (CODEC, CODEC_Q) in ASVspoof5 protocol files.

Prints unique values and counts per split, writes CSV summary.

Usage:
    python scripts/inspect_domains.py
    python scripts/inspect_domains.py --output-csv data/manifests/domain_stats.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from asvspoof5_domain_invariant_cm.data.asvspoof5 import load_protocol
from asvspoof5_domain_invariant_cm.utils.paths import get_asvspoof5_root, get_manifests_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect ASVspoof5 domain labels")
    parser.add_argument(
        "--asvspoof5-root",
        type=Path,
        default=None,
        help="ASVspoof5 data root (overrides ASVSPOOF5_ROOT env var)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path for domain statistics",
    )
    return parser.parse_args()


def inspect_split(df: pd.DataFrame, split_name: str) -> dict:
    """Inspect domain labels in a single split.

    Args:
        df: Protocol DataFrame.
        split_name: Name of the split.

    Returns:
        Dictionary with domain statistics.
    """
    n_samples = len(df)

    # CODEC statistics
    codec_counts = df["codec"].value_counts().to_dict()
    codec_unique = sorted(codec_counts.keys())
    codec_n_unique = len(codec_unique)

    # CODEC_Q statistics
    codec_q_counts = df["codec_q"].value_counts().to_dict()
    codec_q_unique = sorted(codec_q_counts.keys())
    codec_q_n_unique = len(codec_q_unique)

    return {
        "split": split_name,
        "n_samples": n_samples,
        "codec_n_unique": codec_n_unique,
        "codec_unique": codec_unique,
        "codec_counts": codec_counts,
        "codec_q_n_unique": codec_q_n_unique,
        "codec_q_unique": codec_q_unique,
        "codec_q_counts": codec_q_counts,
    }


def print_split_stats(stats: dict) -> None:
    """Print domain statistics for a split."""
    print(f"\n{'='*60}")
    print(f"Split: {stats['split']}")
    print(f"{'='*60}")
    print(f"Total samples: {stats['n_samples']:,}")

    print(f"\nCODEC ({stats['codec_n_unique']} unique values):")
    for codec, count in sorted(stats["codec_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["n_samples"]
        print(f"  {codec:12s}: {count:>8,} ({pct:5.1f}%)")

    print(f"\nCODEC_Q ({stats['codec_q_n_unique']} unique values):")
    for codec_q, count in sorted(stats["codec_q_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["n_samples"]
        print(f"  {codec_q:12s}: {count:>8,} ({pct:5.1f}%)")


def stats_to_csv_rows(all_stats: list[dict]) -> list[dict]:
    """Convert statistics to CSV rows.

    Args:
        all_stats: List of statistics dictionaries.

    Returns:
        List of rows for CSV output.
    """
    rows = []

    for stats in all_stats:
        split = stats["split"]
        n_samples = stats["n_samples"]

        # Add CODEC rows
        for codec, count in stats["codec_counts"].items():
            rows.append({
                "split": split,
                "domain_type": "CODEC",
                "value": codec,
                "count": count,
                "percentage": 100 * count / n_samples,
            })

        # Add CODEC_Q rows
        for codec_q, count in stats["codec_q_counts"].items():
            rows.append({
                "split": split,
                "domain_type": "CODEC_Q",
                "value": codec_q,
                "count": count,
                "percentage": 100 * count / n_samples,
            })

    return rows


def main():
    args = parse_args()

    # Get ASVspoof5 root
    if args.asvspoof5_root:
        asvspoof5_root = args.asvspoof5_root
    else:
        asvspoof5_root = get_asvspoof5_root()

    logger.info(f"ASVspoof5 root: {asvspoof5_root}")

    # Protocol file locations
    protocol_dir = asvspoof5_root / "ASVspoof5_protocols"

    # Check if protocols are in root (already extracted)
    if not protocol_dir.exists():
        protocol_dir = asvspoof5_root

    splits = {
        "train": "ASVspoof5.train.tsv",
        "dev": "ASVspoof5.dev.track_1.tsv",
        "eval": "ASVspoof5.eval.track_1.tsv",
    }

    all_stats = []

    for split_name, protocol_file in splits.items():
        protocol_path = protocol_dir / protocol_file

        if not protocol_path.exists():
            logger.warning(f"Protocol not found: {protocol_path}")
            continue

        logger.info(f"Loading {split_name}: {protocol_path}")
        df = load_protocol(protocol_path)

        stats = inspect_split(df, split_name)
        all_stats.append(stats)
        print_split_stats(stats)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Check if train/dev have codec diversity
    train_stats = next((s for s in all_stats if s["split"] == "train"), None)
    dev_stats = next((s for s in all_stats if s["split"] == "dev"), None)

    if train_stats and train_stats["codec_n_unique"] == 1:
        print("\n⚠️  WARNING: Train set has NO codec diversity (all uncoded)")
        print("   DANN domain discriminator will learn nothing!")
        print("   Synthetic codec augmentation is REQUIRED for meaningful DANN training.")

    if dev_stats and dev_stats["codec_n_unique"] == 1:
        print("\n⚠️  WARNING: Dev set has NO codec diversity (all uncoded)")

    # Write CSV
    if args.output_csv:
        output_path = args.output_csv
    else:
        output_path = get_manifests_dir() / "domain_stats.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = stats_to_csv_rows(all_stats)
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved domain statistics to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
