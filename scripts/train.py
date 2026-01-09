#!/usr/bin/env python3
"""Training entrypoint for ERM and DANN models.

Usage:
    python scripts/train.py --config configs/train/erm.yaml --model configs/model/wavlm_base.yaml
    python scripts/train.py --config configs/train/dann.yaml --model configs/model/wavlm_base.yaml
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
    parser = argparse.ArgumentParser(description="Train deepfake detector")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training config (erm.yaml or dann.yaml)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model config (wavlm_base.yaml, w2v2_base.yaml, etc.)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/data/asvspoof5_track1.yaml"),
        help="Path to data config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ASVspoof 5 Domain-Invariant CM - Training")
    logger.info("=" * 60)
    logger.info(f"Training config: {args.config}")
    logger.info(f"Model config: {args.model}")
    logger.info(f"Data config: {args.data}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Seed: {args.seed}")

    raise NotImplementedError("Training logic not yet implemented.")


if __name__ == "__main__":
    main()
