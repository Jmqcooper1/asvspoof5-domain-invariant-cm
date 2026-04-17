#!/usr/bin/env python3
"""Extract softmax-normalized layer-pooling weights from a trained checkpoint.

Loads only the state-dict tensor; does not reconstruct the model. CPU-only,
fast, works on any node.

Usage:
    python scripts/extract_pool_weights.py \
        --checkpoint runs/wavlm_dann_seed123/checkpoints/best.pt \
        --output results/pool_weights/wavlm_dann_seed123.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_KEY = "backbone.layer_pooling.weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"State-dict key (default: {DEFAULT_KEY})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if args.key not in state:
        candidates = [k for k in state.keys() if "layer_pooling" in k or "pool" in k]
        raise SystemExit(
            f"Key {args.key!r} not in checkpoint. "
            f"Pool-related keys present: {candidates[:20]}"
        )

    raw = state[args.key].detach().cpu().float()
    softmax_w = torch.softmax(raw, dim=0).tolist()

    payload = {
        "checkpoint": str(args.checkpoint),
        "state_key": args.key,
        "num_layers": len(softmax_w),
        "raw_weights": raw.tolist(),
        "softmax_weights": softmax_w,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}  (num_layers={len(softmax_w)})")


if __name__ == "__main__":
    main()
