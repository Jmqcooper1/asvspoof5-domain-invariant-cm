#!/usr/bin/env python3
"""Run layer-wise domain probes to analyze domain leakage.

This script trains linear classifiers on frozen representations from each
layer to predict domain labels (CODEC, CODEC_Q). Lower probe accuracy
indicates more domain-invariant representations.

Usage:
    python scripts/run_probes.py --checkpoint experiments/run_001/best.pt --split dev
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
    parser = argparse.ArgumentParser(description="Run domain probes")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split to probe on",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=1,
        help="ASVspoof track",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to probe: 'all' or comma-separated indices (e.g., '0,3,6,9,11')",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="codec,codec_q",
        help="Domains to probe for (comma-separated)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logistic", "svm"],
        default="logistic",
        help="Probe classifier type",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: checkpoint parent dir / probes)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ASVspoof 5 Domain-Invariant CM - Domain Probes")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Domains: {args.domains}")
    logger.info(f"Classifier: {args.classifier}")

    # TODO: Implement probing logic
    # 1. Load model from checkpoint
    # 2. Load data
    # 3. Extract embeddings from each layer
    # 4. For each layer and each domain:
    #    - Train linear probe (with CV)
    #    - Record accuracy
    # 5. Save results and generate plots

    raise NotImplementedError(
        "Probing logic not yet implemented. "
        "See src/asvspoof5_domain_invariant_cm/analysis/probes.py"
    )


if __name__ == "__main__":
    main()
