"""Path utilities for project directories."""

import os
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root.
    """
    # Try to find pyproject.toml
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current working directory
    return Path.cwd()


def get_data_dir() -> Path:
    """Get data directory.

    Checks ASVSPOOF5_DATA_ROOT environment variable first.

    Returns:
        Path to data directory.
    """
    env_path = os.environ.get("ASVSPOOF5_DATA_ROOT")
    if env_path:
        return Path(env_path)

    return get_project_root() / "data"


def get_experiment_dir(
    name: str = None,
    base_dir: Path = None,
) -> Path:
    """Get or create experiment directory.

    Args:
        name: Experiment name (auto-generated if None).
        base_dir: Base experiments directory.

    Returns:
        Path to experiment directory.
    """
    if base_dir is None:
        base_dir = get_project_root() / "experiments"

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}"

    exp_dir = base_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def get_manifest_path(split: str) -> Path:
    """Get path to manifest file for a split.

    Args:
        split: Data split ('train', 'dev', 'eval').

    Returns:
        Path to manifest file.
    """
    return get_data_dir() / "manifests" / f"{split}.parquet"


def get_audio_dir(split: str) -> Path:
    """Get audio directory for a split.

    Args:
        split: Data split.

    Returns:
        Path to audio directory.
    """
    split_map = {
        "train": "flac_T",
        "dev": "flac_D",
        "eval": "flac_E_eval",
    }

    audio_subdir = split_map.get(split, split)
    return get_data_dir() / "raw" / "asvspoof5" / audio_subdir
