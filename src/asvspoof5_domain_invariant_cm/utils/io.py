"""I/O utilities for checkpoints, metrics, and artifacts."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
    config: Optional[dict] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        metrics: Dictionary of metrics.
        path: Output path.
        config: Optional configuration to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    if config is not None:
        checkpoint["config"] = config

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        checkpoint["git_commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Optional model to load weights into.
        optimizer: Optional optimizer to load state into.
        device: Device to load to.

    Returns:
        Checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    metrics = convert(metrics)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: Path) -> dict:
    """Load metrics from JSON file.

    Args:
        path: Path to metrics file.

    Returns:
        Metrics dictionary.
    """
    with open(path) as f:
        return json.load(f)


def save_config(config: dict, path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_predictions(
    predictions: list[dict],
    path: Path,
) -> None:
    """Save per-utterance predictions to TSV.

    Args:
        predictions: List of prediction dictionaries.
        path: Output path.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(predictions)
    df.to_csv(path, sep="\t", index=False)
