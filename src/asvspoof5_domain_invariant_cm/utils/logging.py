"""Structured logging utilities following wide events philosophy.

This module provides comprehensive logging with:
- Structured JSON logs for queryability
- Wide events with full context per epoch/evaluation
- Experiment context collection (git, hardware, packages)
- Unified interface for console + file + wandb logging
"""

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        if hasattr(record, "event_data"):
            log_data["data"] = record.event_data

        return json.dumps(log_data, default=self._json_serializer)

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serialize numpy/torch types for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


def setup_logging(
    run_dir: Optional[Path] = None,
    level: int = logging.INFO,
    json_output: bool = True,
    console_format: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Configure logging with console and optional JSON file output.

    Args:
        run_dir: Directory for log files. If None, only console logging.
        level: Logging level.
        json_output: Whether to write JSON logs to file.
        console_format: Format string for console output.

    Returns:
        Root logger configured with handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with human-readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(console_format))
    root_logger.addHandler(console_handler)

    # JSON file handler
    if run_dir is not None and json_output:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        json_handler = logging.FileHandler(run_dir / "logs.jsonl", mode="a")
        json_handler.setLevel(level)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)

    return root_logger


def get_git_info() -> dict[str, Any]:
    """Get git repository information."""
    info = {"commit": None, "branch": None, "dirty": None}

    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True
        )
        info["dirty"] = result.returncode != 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return info


def get_hardware_info() -> dict[str, Any]:
    """Get hardware and system information."""
    info = {
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["mps_available"] = True

    return info


def get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    key_packages = [
        "torch",
        "torchaudio",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "wandb",
        "matplotlib",
        "seaborn",
    ]

    for pkg in key_packages:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return packages


def get_experiment_context(config: Optional[dict] = None) -> dict[str, Any]:
    """Collect comprehensive experiment context.

    Args:
        config: Optional configuration dictionary to include hash of.

    Returns:
        Dictionary with full experiment context.
    """
    context = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git": get_git_info(),
        "hardware": get_hardware_info(),
        "packages": get_package_versions(),
        "environment": {
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", "unknown"),
        },
    }

    if config is not None:
        # Create config hash for reproducibility tracking
        config_str = yaml.dump(config, sort_keys=True, default_flow_style=False)
        context["config_hash"] = hashlib.md5(config_str.encode()).hexdigest()[:8]

    return context


def get_gpu_memory_usage() -> Optional[float]:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024**3), 3)
    return None


def get_gpu_utilization() -> Optional[dict[str, float]]:
    """Get GPU utilization stats."""
    if not torch.cuda.is_available():
        return None

    try:
        return {
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
            "max_memory_allocated_gb": round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            ),
        }
    except Exception:
        return None


class ExperimentLogger:
    """Unified logging interface for experiments.

    Provides structured logging to:
    - Console (human-readable)
    - JSON file (structured, queryable)
    - Wandb (optional, for experiment tracking)

    Args:
        run_dir: Directory for saving logs and artifacts.
        run_name: Name of the experiment run.
        config: Configuration dictionary.
        use_wandb: Whether to use wandb logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity (team/username).
        wandb_tags: Tags for wandb run.
        log_code: Whether to log code to wandb.
    """

    def __init__(
        self,
        run_dir: Path,
        run_name: str,
        config: dict,
        use_wandb: bool = False,
        wandb_project: str = "asvspoof5-dann",
        wandb_entity: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        log_code: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Setup logging
        self.logger = setup_logging(run_dir, json_output=True)
        self._logger = logging.getLogger(__name__)

        # Collect and save experiment context
        self.context = get_experiment_context(config)
        self._save_context()

        # Initialize wandb
        if self.use_wandb:
            self._init_wandb(
                project=wandb_project,
                entity=wandb_entity,
                tags=wandb_tags,
                log_code=log_code,
            )

    def _save_context(self) -> None:
        """Save experiment context to JSON file."""
        context_path = self.run_dir / "experiment_context.json"
        with open(context_path, "w") as f:
            json.dump(self.context, f, indent=2, default=str)

    def _init_wandb(
        self,
        project: str,
        entity: Optional[str],
        tags: Optional[list[str]],
        log_code: bool,
    ) -> None:
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            self._logger.warning("Wandb not available, skipping initialization")
            self.use_wandb = False
            return

        try:
            wandb.init(
                project=project,
                entity=entity,
                name=self.run_name,
                config=self.config,
                dir=str(self.run_dir),
                tags=tags,
                save_code=log_code,
            )

            # Log experiment context
            wandb.config.update({"context": self.context}, allow_val_change=True)

            self._logger.info(f"Wandb initialized: {wandb.run.url}")
        except Exception as e:
            self._logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False

    def log_wide_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a wide event with full context.

        Args:
            event_type: Type of event (e.g., 'epoch_complete', 'evaluation_complete')
            data: Event data dictionary.
        """
        # Add standard fields
        event = {
            "event_type": event_type,
            "run_id": self.run_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **data,
        }

        # Add GPU memory if available
        gpu_mem = get_gpu_memory_usage()
        if gpu_mem is not None:
            event["gpu_memory_gb"] = gpu_mem

        # Log to JSON file via custom record
        record = logging.LogRecord(
            name=__name__,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"[{event_type}]",
            args=(),
            exc_info=None,
        )
        record.event_type = event_type
        record.event_data = data

        # Write directly to JSON handler
        for handler in self.logger.handlers:
            if isinstance(handler.formatter, JsonFormatter):
                handler.emit(record)

        # Also log human-readable summary
        self._logger.info(f"[{event_type}] {self._summarize_event(event)}")

        # Log to wandb
        if self.use_wandb:
            self._log_to_wandb(event_type, event)

    def _summarize_event(self, event: dict) -> str:
        """Create human-readable summary of event."""
        event_type = event.get("event_type", "unknown")

        if event_type == "epoch_complete":
            epoch = event.get("epoch", "?")
            train_loss = event.get("train", {}).get("loss", "?")
            val_eer = event.get("val", {}).get("eer", "?")
            if isinstance(train_loss, float):
                train_loss = f"{train_loss:.4f}"
            if isinstance(val_eer, float):
                val_eer = f"{val_eer:.4f}"
            return f"epoch={epoch}, train_loss={train_loss}, val_eer={val_eer}"

        elif event_type == "evaluation_complete":
            split = event.get("split", "?")
            eer = event.get("metrics", {}).get("eer", "?")
            if isinstance(eer, float):
                eer = f"{eer:.4f}"
            return f"split={split}, eer={eer}"

        elif event_type == "experiment_start":
            method = event.get("method", "?")
            backbone = event.get("backbone", "?")
            return f"method={method}, backbone={backbone}"

        return json.dumps(event, default=str)[:200]

    def _log_to_wandb(self, event_type: str, event: dict) -> None:
        """Log event to wandb."""
        if not self.use_wandb:
            return

        # Flatten nested dicts for wandb
        flat = {}

        if event_type == "epoch_complete":
            epoch = event.get("epoch", 0)
            flat["epoch"] = epoch

            # Training metrics
            train = event.get("train", {})
            for k, v in train.items():
                if isinstance(v, (int, float)):
                    flat[f"train/{k}"] = v

            # Validation metrics
            val = event.get("val", {})
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    flat[f"val/{k}"] = v

            # Other scalars
            for k in ["learning_rate", "lambda_domain", "grad_norm_mean", "grad_clips", "gpu_memory_gb"]:
                if k in event and isinstance(event[k], (int, float)):
                    flat[k] = event[k]

            # Layer weights as histogram
            if "layer_weights" in event:
                flat["layer_weights"] = wandb.Histogram(event["layer_weights"])

            wandb.log(flat, step=epoch)

        elif event_type == "evaluation_complete":
            metrics = event.get("metrics", {})
            split = event.get("split", "eval")

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    flat[f"eval/{split}/{k}"] = v

            wandb.log(flat)

        elif event_type == "experiment_start":
            # Just log config update, already done in init
            pass

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to wandb and console.

        Args:
            metrics: Dictionary of metrics.
            step: Optional step number.
        """
        # Console log
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        )
        self._logger.info(f"Metrics: {metrics_str}")

        # Wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_table(
        self,
        name: str,
        dataframe,
        step: Optional[int] = None,
    ) -> None:
        """Log a pandas DataFrame as wandb.Table.

        Args:
            name: Table name.
            dataframe: Pandas DataFrame.
            step: Optional step number.
        """
        if self.use_wandb:
            table = wandb.Table(dataframe=dataframe)
            wandb.log({name: table}, step=step)

        # Also save as CSV
        csv_path = self.run_dir / "tables" / f"{name}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)
        self._logger.info(f"Saved table: {csv_path}")

    def log_image(
        self,
        name: str,
        image_path: Path,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image to wandb.

        Args:
            name: Image name/key.
            image_path: Path to image file.
            step: Optional step number.
            caption: Optional image caption.
        """
        if self.use_wandb:
            wandb.log(
                {name: wandb.Image(str(image_path), caption=caption)},
                step=step,
            )
        self._logger.info(f"Logged image: {name} ({image_path})")

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: Path,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an artifact to wandb.

        Args:
            name: Artifact name.
            artifact_type: Type (e.g., 'model', 'dataset').
            path: Path to artifact file/directory.
            metadata: Optional metadata dict.
        """
        if self.use_wandb:
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            if Path(path).is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
            self._logger.info(f"Logged artifact: {name} ({artifact_type})")

    def log_model_summary(
        self,
        model: torch.nn.Module,
        watch_gradients: bool = False,
    ) -> None:
        """Log model architecture summary.

        Args:
            model: PyTorch model.
            watch_gradients: Whether to track gradients with wandb.watch.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        summary = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
        }

        self._logger.info(
            f"Model: {total_params:,} params ({trainable_params:,} trainable)"
        )

        if self.use_wandb:
            wandb.config.update({"model_summary": summary}, allow_val_change=True)
            if watch_gradients:
                wandb.watch(model, log="gradients", log_freq=100)

    def log_dataset_stats(
        self,
        train_size: int,
        val_size: int,
        class_distribution: Optional[dict] = None,
        domain_distribution: Optional[dict] = None,
    ) -> None:
        """Log dataset statistics.

        Args:
            train_size: Number of training samples.
            val_size: Number of validation samples.
            class_distribution: Optional class label counts.
            domain_distribution: Optional domain label counts.
        """
        stats = {
            "train_samples": train_size,
            "val_samples": val_size,
        }

        if class_distribution:
            stats["class_distribution"] = class_distribution
        if domain_distribution:
            stats["domain_distribution"] = domain_distribution

        self._logger.info(f"Dataset: train={train_size}, val={val_size}")

        if self.use_wandb:
            wandb.config.update({"dataset_stats": stats}, allow_val_change=True)

        # Save to file
        stats_path = self.run_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def set_summary(self, metrics: dict[str, Any]) -> None:
        """Set wandb summary metrics (final/best results).

        Args:
            metrics: Dictionary of summary metrics.
        """
        if self.use_wandb:
            for k, v in metrics.items():
                wandb.run.summary[k] = v

    def finish(self) -> None:
        """Finalize logging and cleanup."""
        self._logger.info(f"Experiment complete. Logs saved to: {self.run_dir}")
        if self.use_wandb:
            wandb.finish()


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm across all parameters.

    Args:
        model: PyTorch model.

    Returns:
        Total gradient L2 norm.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm**0.5


def check_for_nan_grads(model: torch.nn.Module) -> bool:
    """Check if any gradients contain NaN values.

    Args:
        model: PyTorch model.

    Returns:
        True if any NaN gradients found.
    """
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            return True
    return False
