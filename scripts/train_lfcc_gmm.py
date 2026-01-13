#!/usr/bin/env python3
"""Train GMM classifier on LFCC features.

This provides a classical baseline using LFCC + GMM backend.

Usage:
    python scripts/train_lfcc_gmm.py
    python scripts/train_lfcc_gmm.py --n-components 64
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from asvspoof5_domain_invariant_cm.evaluation import (
    compute_eer,
    compute_min_dcf,
    evaluate_per_domain,
)
from asvspoof5_domain_invariant_cm.utils import get_features_dir, get_run_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LFCC-GMM classifier")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Directory with LFCC features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=32,
        help="Number of GMM components",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_split(features_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata for a split."""
    embeddings = np.load(features_dir / f"{split}.npy")
    metadata = pd.read_csv(features_dir / f"{split}_metadata.csv")
    return embeddings, metadata


def main():
    args = parse_args()

    np.random.seed(args.seed)

    # Features directory
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = get_features_dir("lfcc")

    logger.info(f"Features directory: {features_dir}")

    # Check if features exist
    if not (features_dir / "train.npy").exists():
        logger.error(
            "LFCC features not found. Run extract_lfcc.py first:\n"
            "  python scripts/extract_lfcc.py --split train\n"
            "  python scripts/extract_lfcc.py --split dev"
        )
        return 1

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_run_dir(f"lfcc_gmm_{args.n_components}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading training data...")
    train_embeddings, train_metadata = load_split(features_dir, "train")
    train_labels = train_metadata["y_task"].values

    logger.info(f"Training samples: {len(train_embeddings)}")
    logger.info(f"Feature dimension: {train_embeddings.shape[1]}")

    logger.info("Loading validation data...")
    val_embeddings, val_metadata = load_split(features_dir, "dev")
    val_labels = val_metadata["y_task"].values

    logger.info(f"Validation samples: {len(val_embeddings)}")

    # Standardize features
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_embeddings)
    val_scaled = scaler.transform(val_embeddings)

    # Separate by class
    train_bonafide = train_scaled[train_labels == 0]
    train_spoof = train_scaled[train_labels == 1]

    logger.info(f"Bonafide samples: {len(train_bonafide)}")
    logger.info(f"Spoof samples: {len(train_spoof)}")

    # Train GMMs
    logger.info(f"Training GMM with {args.n_components} components...")

    gmm_bonafide = GaussianMixture(
        n_components=args.n_components,
        covariance_type="diag",
        max_iter=200,
        random_state=args.seed,
    )
    gmm_bonafide.fit(train_bonafide)
    logger.info("  Bonafide GMM trained")

    gmm_spoof = GaussianMixture(
        n_components=args.n_components,
        covariance_type="diag",
        max_iter=200,
        random_state=args.seed,
    )
    gmm_spoof.fit(train_spoof)
    logger.info("  Spoof GMM trained")

    # Compute scores
    # Score = log P(x | bonafide) - log P(x | spoof)
    # Higher score = more likely bonafide (class 0)
    val_log_prob_bonafide = gmm_bonafide.score_samples(val_scaled)
    val_log_prob_spoof = gmm_spoof.score_samples(val_scaled)
    val_scores = val_log_prob_bonafide - val_log_prob_spoof

    # Predictions: positive score = bonafide
    val_preds = (val_scores < 0).astype(int)  # spoof if score < 0

    # Compute metrics
    eer, eer_threshold = compute_eer(val_scores, val_labels)
    min_dcf = compute_min_dcf(val_scores, val_labels)
    accuracy = (val_preds == val_labels).mean()

    logger.info("=" * 60)
    logger.info("Validation Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  EER: {eer:.4f}")
    logger.info(f"  minDCF: {min_dcf:.4f}")
    logger.info("=" * 60)

    # Per-domain evaluation
    val_df = val_metadata.copy()
    val_df["score"] = val_scores
    val_df["prediction"] = val_preds

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    for domain_col in ["codec", "codec_q"]:
        if domain_col in val_df.columns:
            domain_results = evaluate_per_domain(
                val_df, "score", "y_task", domain_col
            )
            domain_results.to_csv(tables_dir / f"metrics_by_{domain_col}.csv", index=False)
            logger.info(f"\n{domain_col.upper()} breakdown:")
            logger.info(domain_results.to_string(index=False))

    # Save results
    results = {
        "n_components": args.n_components,
        "feature_dim": int(train_embeddings.shape[1]),
        "train_samples": len(train_embeddings),
        "train_bonafide": len(train_bonafide),
        "train_spoof": len(train_spoof),
        "val_samples": len(val_embeddings),
        "val_accuracy": float(accuracy),
        "val_eer": float(eer),
        "val_min_dcf": float(min_dcf),
        "eer_threshold": float(eer_threshold),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    val_df.to_csv(output_dir / "predictions_dev.tsv", sep="\t", index=False)

    # Save models
    import joblib
    joblib.dump({
        "gmm_bonafide": gmm_bonafide,
        "gmm_spoof": gmm_spoof,
        "scaler": scaler,
    }, output_dir / "model.joblib")

    logger.info(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
