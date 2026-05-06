#!/usr/bin/env python3
"""Extract softmax pool weights from every checkpoint in a run dir.

Walks <run_dir>/checkpoints/ and dumps one JSON file per run that maps
checkpoint label (epoch index, "best", "last") to softmax-normalized pool
weights and the lambda_grl value at that checkpoint's epoch (parsed from
logs.jsonl). Run on cluster login node — CPU only.

Usage:
    python scripts/extract_pool_weights_trajectory.py \\
        --run-dir /gpfs/work5/0/prjs1904/runs/wavlm_dann_seed42_v2_1e2d5c7 \\
        --output  results/pool_weights_trajectory/wavlm_dann_seed42_v2.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


DEFAULT_KEY = "backbone.layer_pooling.weights"
LAMBDA_RE = re.compile(r"Epoch\s+(\d+):\s*lambda_grl=([\d.eE+-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-key", default=DEFAULT_KEY)
    return parser.parse_args()


def epoch_for_checkpoint(name: str, ckpt_payload: dict) -> int | None:
    """Return the epoch index this checkpoint corresponds to.

    Tries the embedded "epoch" key first; falls back to parsing the
    file name (`epoch_5.pt` -> 5).
    """
    if "epoch" in ckpt_payload and isinstance(ckpt_payload["epoch"], int):
        return ckpt_payload["epoch"]
    m = re.match(r"epoch_(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def lambdas_per_epoch(logs_path: Path) -> dict[int, float]:
    """Latest lambda_grl logged per epoch (logs may contain restart attempts)."""
    out: dict[int, float] = {}
    if not logs_path.exists():
        return out
    with logs_path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message", "")
            match = LAMBDA_RE.search(msg)
            if match:
                out[int(match.group(1))] = float(match.group(2))
    return out


def extract_one(checkpoint_path: Path, state_key: str) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if state_key not in state:
        raise SystemExit(f"Key {state_key!r} not found in {checkpoint_path.name}")
    raw = state[state_key].detach().cpu().float()
    softmax_w = torch.softmax(raw, dim=0).tolist()
    return {
        "raw_weights": raw.tolist(),
        "softmax_weights": softmax_w,
        "embedded_epoch": payload.get("epoch"),
    }


def main() -> None:
    args = parse_args()
    ckpt_dir = args.run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"No checkpoints/ subdir under {args.run_dir}")

    lambdas = lambdas_per_epoch(args.run_dir / "logs.jsonl")

    entries: dict[str, dict] = {}
    for path in sorted(ckpt_dir.glob("*.pt")):
        label = path.stem  # epoch_5, best, last, ...
        try:
            data = extract_one(path, args.state_key)
        except Exception as e:  # noqa: BLE001
            entries[label] = {"error": str(e)}
            continue
        # Resolve which training epoch this checkpoint corresponds to
        ep = data["embedded_epoch"]
        if ep is None:
            ep = epoch_for_checkpoint(label, {})
        data["epoch"] = ep
        data["lambda_grl"] = lambdas.get(ep) if ep is not None else None
        entries[label] = data
        print(
            f"  {label:<12}  epoch={ep!s:<4}  "
            f"lambda={data['lambda_grl']!s:<6}  "
            f"top3=L0:{data['softmax_weights'][0]*100:.2f}% "
            f"L1:{data['softmax_weights'][1]*100:.2f}% "
            f"L2:{data['softmax_weights'][2]*100:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "state_key": args.state_key,
        "entries": entries,
    }, indent=2) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
