#!/usr/bin/env python3
"""Train classifier on TRILLsson embeddings.

Usage:
    python scripts/train_trillsson.py --classifier logistic
    python scripts/train_trillsson.py --classifier mlp
    python scripts/train_trillsson.py --classifier logistic --wandb
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from asvspoof5_domain_invariant_cm.evaluation import (
    compute_eer,
    compute_min_dcf,
    evaluate_per_domain,
)
from asvspoof5_domain_invariant_cm.models.nonsemantic import (
    predict_sklearn,
    train_sklearn_classifier,
)
from asvspoof5_domain_invariant_cm.utils import get_features_dir, get_run_dir

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRILLsson classifier")
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logistic", "mlp"],
        default="logistic",
        help="Classifier type",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Directory with TRILLsson features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-dann",
        help="Wandb project name",
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
        features_dir = get_features_dir("trillsson")

    logger.info(f"Features directory: {features_dir}")

    # Check if features exist
    if not (features_dir / "train.npy").exists():
        logger.error(
            "TRILLsson features not found. Run extract_trillsson.py first:\n"
            "  python scripts/extract_trillsson.py --split train\n"
            "  python scripts/extract_trillsson.py --split dev"
        )
        return 1

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = get_run_dir(f"trillsson_{args.classifier}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading training data...")
    train_embeddings, train_metadata = load_split(features_dir, "train")
    train_labels = train_metadata["y_task"].values

    logger.info(f"Training samples: {len(train_embeddings)}")
    logger.info(f"Embedding dimension: {train_embeddings.shape[1]}")

    logger.info("Loading validation data...")
    val_embeddings, val_metadata = load_split(features_dir, "dev")
    val_labels = val_metadata["y_task"].values

    logger.info(f"Validation samples: {len(val_embeddings)}")

    # Train classifier
    logger.info(f"Training {args.classifier} classifier...")
    clf, scaler = train_sklearn_classifier(
        train_embeddings,
        train_labels,
        classifier_type=args.classifier,
        random_state=args.seed,
    )

    # Evaluate on validation
    logger.info("Evaluating on validation set...")
    val_preds, val_probs = predict_sklearn(clf, scaler, val_embeddings)

    # Score convention: higher = more bonafide (class 0)
    val_scores = val_probs[:, 0]

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
        "classifier": args.classifier,
        "embedding_dim": int(train_embeddings.shape[1]),
        "train_samples": len(train_embeddings),
        "val_samples": len(val_embeddings),
        "val_accuracy": float(accuracy),
        "val_eer": float(eer),
        "val_min_dcf": float(min_dcf),
        "eer_threshold": float(eer_threshold),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Wandb logging
    use_wandb = args.wandb and WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"trillsson_{args.classifier}",
                config={
                    "method": "trillsson",
                    "classifier": args.classifier,
                    "embedding_dim": int(train_embeddings.shape[1]),
                    "seed": args.seed,
                },
                tags=["baseline", "trillsson", args.classifier],
            )
            wandb.log({
                "val/accuracy": accuracy,
                "val/eer": eer,
                "val/min_dcf": min_dcf,
                "train/n_samples": len(train_embeddings),
            })
            wandb.run.summary["val_eer"] = eer
            wandb.run.summary["val_min_dcf"] = min_dcf
            wandb.finish()
            logger.info("Logged results to Wandb")
        except Exception as e:
            logger.warning(f"Wandb logging failed: {e}")

    # Save predictions
    val_df.to_csv(output_dir / "predictions_dev.tsv", sep="\t", index=False)

    # Save model (sklearn)
    import joblib
    joblib.dump({"classifier": clf, "scaler": scaler}, output_dir / "model.joblib")

    logger.info(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
