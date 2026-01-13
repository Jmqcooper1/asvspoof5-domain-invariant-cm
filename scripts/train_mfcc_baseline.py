#!/usr/bin/env python3
"""Train MFCC baseline classifier.

Uses pre-extracted MFCC features (from extract_mfcc.py) to train either:
- Logistic regression (fast, no GPU)
- Small MLP (optional)

Usage:
    # Train logistic regression baseline
    python scripts/train_mfcc_baseline.py --method logreg

    # Train MLP baseline
    python scripts/train_mfcc_baseline.py --method mlp --epochs 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from asvspoof5_domain_invariant_cm.models.mfcc_baseline import (
    compute_mfcc_metrics,
    train_logreg_baseline,
)
from asvspoof5_domain_invariant_cm.utils.paths import get_features_dir, get_run_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MFCC baseline")
    parser.add_argument(
        "--method",
        type=str,
        choices=["logreg", "mlp"],
        default="logreg",
        help="Training method",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Features directory (default: data/features/mfcc)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter (logreg)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Max iterations (logreg)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (mlp)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension (mlp)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (mlp)",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Compute per-domain breakdown",
    )
    return parser.parse_args()


def load_features(features_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load features and metadata for a split.

    Args:
        features_dir: Features directory.
        split: Split name.

    Returns:
        Tuple of (features array, metadata DataFrame).
    """
    features_path = features_dir / f"{split}.npy"
    metadata_path = features_dir / f"{split}_metadata.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)

    return features, metadata


def compute_per_domain_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    domain_col: str,
) -> dict:
    """Compute metrics per domain value.

    Args:
        scores: Prediction scores.
        labels: Ground truth labels.
        metadata: Metadata DataFrame.
        domain_col: Domain column name.

    Returns:
        Dictionary mapping domain value to metrics.
    """
    results = {}

    for domain_val in metadata[domain_col].unique():
        mask = metadata[domain_col] == domain_val
        if mask.sum() < 10:
            continue

        domain_scores = scores[mask]
        domain_labels = labels[mask]

        # Check if both classes present
        if len(np.unique(domain_labels)) < 2:
            continue

        try:
            metrics = compute_mfcc_metrics(domain_scores, domain_labels)
            results[str(domain_val)] = {
                "n_samples": int(mask.sum()),
                "eer": float(metrics["eer"]),
                "min_dcf": float(metrics["min_dcf"]),
            }
        except Exception as e:
            logger.warning(f"Error computing metrics for {domain_col}={domain_val}: {e}")

    return results


def train_logreg(args, features_dir: Path, output_dir: Path) -> dict:
    """Train logistic regression baseline.

    Args:
        args: Command-line arguments.
        features_dir: Features directory.
        output_dir: Output directory.

    Returns:
        Results dictionary.
    """
    logger.info("Training logistic regression baseline")

    # Load features
    train_features, train_meta = load_features(features_dir, "train")
    val_features, val_meta = load_features(features_dir, "dev")

    train_labels = train_meta["y_task"].values
    val_labels = val_meta["y_task"].values

    logger.info(f"Train: {len(train_features)} samples, {train_features.shape[1]} features")
    logger.info(f"Val: {len(val_features)} samples")

    # Train
    results = train_logreg_baseline(
        train_features,
        train_labels,
        val_features,
        val_labels,
        max_iter=args.max_iter,
        C=args.C,
    )

    # Compute metrics
    train_metrics = compute_mfcc_metrics(results["train_probs"], train_labels)
    val_metrics = compute_mfcc_metrics(results["val_probs"], val_labels)

    logger.info(f"Train: acc={results['train_acc']:.4f}, EER={train_metrics['eer']:.4f}")
    logger.info(f"Val: acc={results['val_acc']:.4f}, EER={val_metrics['eer']:.4f}, minDCF={val_metrics['min_dcf']:.4f}")

    output = {
        "method": "logreg",
        "config": {
            "C": args.C,
            "max_iter": args.max_iter,
        },
        "train": {
            "acc": float(results["train_acc"]),
            "eer": float(train_metrics["eer"]),
            "min_dcf": float(train_metrics["min_dcf"]),
            "n_samples": len(train_features),
        },
        "val": {
            "acc": float(results["val_acc"]),
            "eer": float(val_metrics["eer"]),
            "min_dcf": float(val_metrics["min_dcf"]),
            "n_samples": len(val_features),
        },
    }

    # Per-domain breakdown
    if args.per_domain:
        logger.info("Computing per-domain breakdown...")

        output["val_per_codec"] = compute_per_domain_metrics(
            results["val_probs"],
            val_labels,
            val_meta,
            "codec",
        )

        output["val_per_codec_q"] = compute_per_domain_metrics(
            results["val_probs"],
            val_labels,
            val_meta,
            "codec_q",
        )

        # Print per-codec EER
        if output["val_per_codec"]:
            logger.info("Per-CODEC EER:")
            for codec, metrics in sorted(output["val_per_codec"].items()):
                logger.info(f"  {codec}: EER={metrics['eer']:.4f} (n={metrics['n_samples']})")

    # Save model
    import joblib
    model_path = output_dir / "logreg_model.joblib"
    joblib.dump({
        "model": results["model"],
        "scaler": results["scaler"],
    }, model_path)
    logger.info(f"Saved model: {model_path}")

    return output


def main():
    args = parse_args()

    # Features directory
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = get_features_dir("mfcc")

    logger.info(f"Features directory: {features_dir}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_run_dir(f"mfcc_{args.method}_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Train
    if args.method == "logreg":
        results = train_logreg(args, features_dir, output_dir)
    else:
        logger.error(f"Method {args.method} not implemented yet")
        return 1

    # Save results
    results_path = output_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results: {results_path}")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
