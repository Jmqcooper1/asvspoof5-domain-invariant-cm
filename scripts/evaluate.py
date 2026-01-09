#!/usr/bin/env python3
"""Evaluation entrypoint for trained models.

Usage:
    python scripts/evaluate.py --checkpoint experiments/run_001/best.pt --split dev
    python scripts/evaluate.py --checkpoint experiments/run_001/best.pt --split eval --track 1
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "eval"],
        default="dev",
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=1,
        help="ASVspoof track (1 for stand-alone detection)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval/track1_metrics.yaml"),
        help="Path to evaluation config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: checkpoint parent dir)",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Compute per-domain (CODEC, CODEC_Q) metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ASVspoof 5 Domain-Invariant CM - Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Track: {args.track}")
    logger.info(f"Per-domain: {args.per_domain}")

    # TODO: Implement evaluation logic
    # 1. Load model from checkpoint
    # 2. Load evaluation data
    # 3. Run inference (get scores)
    # 4. Compute metrics (EER, minDCF, etc.)
    # 5. Per-domain breakdown (if requested)
    # 6. Save predictions and metrics

    raise NotImplementedError(
        "Evaluation logic not yet implemented. "
        "See src/asvspoof5_domain_invariant_cm/evaluation/ for module structure."
    )


if __name__ == "__main__":
    main()
