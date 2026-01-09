#!/usr/bin/env python3
"""Run activation patching experiments.

This script performs limited activation patching to test causal effects
of domain-heavy components on detection and domain leakage.

Usage:
    python scripts/run_patching.py \
        --source experiments/dann_run/best.pt \
        --target experiments/erm_run/best.pt \
        --split dev
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
    parser = argparse.ArgumentParser(description="Run activation patching")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source model checkpoint (e.g., DANN model)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target model checkpoint (e.g., ERM model)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split to run patching on",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="projection",
        help="Layers to patch: 'projection', 'head', or layer indices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for patching experiment",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ASVspoof 5 Domain-Invariant CM - Activation Patching")
    logger.info("=" * 60)
    logger.info(f"Source model: {args.source}")
    logger.info(f"Target model: {args.target}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Layers to patch: {args.layers}")

    # TODO: Implement patching logic
    # 1. Load source and target models
    # 2. Load data
    # 3. For each sample:
    #    a. Run forward pass on both models
    #    b. Patch activations from source into target at specified layers
    #    c. Record predictions and domain probe outputs
    # 4. Compare:
    #    - Task performance before/after patching
    #    - Domain probe accuracy before/after patching
    # 5. Save results

    raise NotImplementedError(
        "Patching logic not yet implemented. "
        "See src/asvspoof5_domain_invariant_cm/analysis/patching.py"
    )


if __name__ == "__main__":
    main()
