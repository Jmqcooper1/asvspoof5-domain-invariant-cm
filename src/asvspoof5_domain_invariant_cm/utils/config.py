"""Configuration loading and management."""

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict:
    """Load YAML configuration file.

    Args:
        path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(*configs: dict) -> dict:
    """Recursively merge multiple configuration dictionaries.

    Later configs override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        Merged configuration.
    """
    result = {}

    for config in configs:
        for key, value in config.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value

    return result


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, use deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get computation device.

    Args:
        device: Device string ('cuda', 'cpu', 'mps', or None for auto).

    Returns:
        torch.device object.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten nested config dictionary.

    Args:
        config: Nested configuration dictionary.
        prefix: Key prefix for flattening.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    flat = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value

    return flat


def unflatten_config(flat_config: dict) -> dict:
    """Unflatten a flat config dictionary.

    Args:
        flat_config: Flat dictionary with dot-separated keys.

    Returns:
        Nested dictionary.
    """
    result = {}

    for key, value in flat_config.items():
        parts = key.split(".")
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result
