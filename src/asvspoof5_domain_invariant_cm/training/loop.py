"""Training and validation loops with comprehensive logging."""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.io import save_checkpoint, save_config, save_metrics
from ..utils.logging import (
    ExperimentLogger,
    check_for_nan_grads,
    compute_grad_norm,
    get_gpu_memory_usage,
)

logger = logging.getLogger(__name__)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    method: str = "erm",
    log_interval: int = 50,
    batch_sample_rate: float = 0.0,
    track_gradients: bool = True,
    exp_logger: Optional[ExperimentLogger] = None,
    global_step_start: int = 0,
    log_step_interval: int = 100,
) -> tuple[dict, int]:
    """Train for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Device.
        scheduler: Optional LR scheduler.
        scaler: Optional gradient scaler for AMP.
        gradient_clip: Gradient clipping value.
        method: Training method ('erm' or 'dann').
        log_interval: Logging interval for progress bar.
        batch_sample_rate: Rate of batches to log detailed info (0.0-1.0).
        track_gradients: Whether to track gradient norms.
        exp_logger: Optional ExperimentLogger for step-level wandb logging.
        global_step_start: Starting global step for this epoch.
        log_step_interval: Interval for step-level wandb logging.

    Returns:
        Tuple of (metrics dict, final global step).
    """
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_codec_loss = 0.0
    total_codec_q_loss = 0.0
    total_task_acc = 0.0
    total_codec_acc = 0.0
    total_codec_q_acc = 0.0
    num_batches = 0

    # Gradient tracking
    grad_norms = []
    grad_clips_count = 0
    nan_grad_count = 0

    # Augmentation rate tracking (for DANN)
    total_aug_samples = 0
    total_non_none_samples = 0

    # Batch-level logging samples
    batch_samples = []

    # Global step tracking
    global_step = global_step_start

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch_start_time = time.time()

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"].to(device)

        # Use augmented domain labels when available (for DANN with synthetic augmentation)
        if "y_codec_aug" in batch and batch["y_codec_aug"] is not None:
            y_codec = batch["y_codec_aug"].to(device)
            y_codec_q = batch["y_codec_q_aug"].to(device)
        else:
            y_codec = batch["y_codec"].to(device)
            y_codec_q = batch["y_codec_q"].to(device)

        optimizer.zero_grad()

        # Forward pass
        use_amp = scaler is not None
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=use_amp):
            outputs = model(waveform, attention_mask, lengths)

            if method == "dann":
                # Fail-fast: DANN requires domain diversity in early training
                # Check first 10 batches to catch wiring bugs immediately
                if batch_idx < 10:
                    unique_codecs = y_codec.unique().numel()
                    if unique_codecs < 2:
                        raise RuntimeError(
                            f"DANN requires domain diversity but batch {batch_idx} has only "
                            f"{unique_codecs} unique codec(s). Check: augmentor wired? "
                            f"ffmpeg available? supported_codecs>=2? codec_prob>0?"
                        )

                losses = loss_fn(
                    outputs["task_logits"],
                    outputs["codec_logits"],
                    outputs["codec_q_logits"],
                    y_task,
                    y_codec,
                    y_codec_q,
                )
            else:
                losses = loss_fn(outputs["task_logits"], y_task)

        loss = losses["total_loss"]

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Track gradients before clipping
            if track_gradients:
                grad_norm = compute_grad_norm(model)
                grad_norms.append(grad_norm)

                # Check for NaN gradients
                if check_for_nan_grads(model):
                    nan_grad_count += 1
                    logger.warning(f"NaN gradient detected at batch {batch_idx}")

            # Clip gradients
            orig_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if orig_norm > gradient_clip:
                grad_clips_count += 1

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Track gradients before clipping
            if track_gradients:
                grad_norm = compute_grad_norm(model)
                grad_norms.append(grad_norm)

                if check_for_nan_grads(model):
                    nan_grad_count += 1
                    logger.warning(f"NaN gradient detected at batch {batch_idx}")

            orig_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if orig_norm > gradient_clip:
                grad_clips_count += 1

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Increment global step
        global_step += 1

        # Compute accuracies
        task_acc = compute_accuracy(outputs["task_logits"], y_task)

        # Accumulate metrics
        total_loss += losses["total_loss"].item()
        total_task_loss += losses["task_loss"].item()
        total_task_acc += task_acc
        num_batches += 1

        if method == "dann":
            total_codec_loss += losses["codec_loss"].item()
            total_codec_q_loss += losses["codec_q_loss"].item()
            codec_acc = compute_accuracy(outputs["codec_logits"], y_codec)
            codec_q_acc = compute_accuracy(outputs["codec_q_logits"], y_codec_q)
            total_codec_acc += codec_acc
            total_codec_q_acc += codec_q_acc

            # Track augmentation rate (assuming 0 = NONE in synthetic vocab)
            total_aug_samples += y_codec.numel()
            total_non_none_samples += (y_codec != 0).sum().item()

            # Log augmentation rate periodically and fail if too low
            if batch_idx > 0 and batch_idx % 100 == 0:
                aug_rate = total_non_none_samples / max(total_aug_samples, 1)
                logger.info(
                    f"Step {batch_idx}: cumulative augmentation rate = {aug_rate:.1%} "
                    f"({total_non_none_samples}/{total_aug_samples} samples coded)"
                )

                # Fail if augmentation rate is near zero after sufficient steps
                if batch_idx >= 500 and aug_rate < 0.05:
                    raise RuntimeError(
                        f"Augmentation rate {aug_rate:.1%} < 5% after {batch_idx} steps. "
                        f"DANN requires domain diversity. Check codec_prob and ffmpeg codec support."
                    )

        # Sample batches for detailed logging
        if batch_sample_rate > 0 and random.random() < batch_sample_rate:
            batch_duration = time.time() - batch_start_time
            batch_event = {
                "batch_idx": batch_idx,
                "loss": losses["total_loss"].item(),
                "task_loss": losses["task_loss"].item(),
                "task_acc": task_acc,
                "grad_norm": grad_norms[-1] if grad_norms else None,
                "batch_duration_sec": batch_duration,
                "batch_size": waveform.shape[0],
            }
            if method == "dann":
                batch_event["codec_loss"] = losses["codec_loss"].item()
                batch_event["codec_q_loss"] = losses["codec_q_loss"].item()
                batch_event["codec_acc"] = codec_acc
                batch_event["codec_q_acc"] = codec_q_acc
            batch_samples.append(batch_event)

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "task_acc": f"{total_task_acc / num_batches:.4f}",
            })

        # Step-level wandb logging
        if exp_logger is not None and global_step % log_step_interval == 0:
            step_metrics = {
                "train/step_loss": losses["total_loss"].item(),
                "train/step_task_loss": losses["task_loss"].item(),
                "train/step_task_acc": task_acc,
            }
            if method == "dann":
                step_metrics["train/step_codec_loss"] = losses["codec_loss"].item()
                step_metrics["train/step_codec_q_loss"] = losses["codec_q_loss"].item()
                step_metrics["train/step_codec_acc"] = codec_acc
                step_metrics["train/step_codec_q_acc"] = codec_q_acc
            if track_gradients and grad_norms:
                step_metrics["train/step_grad_norm"] = grad_norms[-1]
            exp_logger.log_step_metrics(step_metrics, step=global_step)

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "task_acc": total_task_acc / num_batches,
    }

    if method == "dann":
        metrics["codec_loss"] = total_codec_loss / num_batches
        metrics["codec_q_loss"] = total_codec_q_loss / num_batches
        metrics["codec_acc"] = total_codec_acc / num_batches
        metrics["codec_q_acc"] = total_codec_q_acc / num_batches
        # Augmentation rate metric
        if total_aug_samples > 0:
            metrics["aug_rate"] = total_non_none_samples / total_aug_samples

    # Gradient statistics
    if track_gradients and grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(grad_norms))
        metrics["grad_norm_max"] = float(np.max(grad_norms))
        metrics["grad_clips"] = grad_clips_count
        metrics["nan_grads"] = nan_grad_count

    # Include batch samples for wide event logging
    if batch_samples:
        metrics["_batch_samples"] = batch_samples

    return metrics, global_step


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    method: str = "erm",
    compute_domain_breakdown: bool = False,
    codec_vocab: Optional[dict] = None,
    codec_q_vocab: Optional[dict] = None,
) -> dict:
    """Validate for one epoch.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: Device.
        method: Training method ('erm' or 'dann').
        compute_domain_breakdown: Whether to compute per-domain metrics.
        codec_vocab: CODEC vocabulary (for domain breakdown).
        codec_q_vocab: CODEC_Q vocabulary (for domain breakdown).

    Returns:
        Dictionary of average metrics for the epoch.
    """
    model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_codec_loss = 0.0
    total_codec_q_loss = 0.0
    total_task_acc = 0.0
    total_codec_acc = 0.0
    total_codec_q_acc = 0.0
    num_batches = 0

    all_scores = []
    all_labels = []
    all_codec_labels = []
    all_codec_q_labels = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"].to(device)
        y_codec = batch["y_codec"].to(device)
        y_codec_q = batch["y_codec_q"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        if method == "dann":
            losses = loss_fn(
                outputs["task_logits"],
                outputs["codec_logits"],
                outputs["codec_q_logits"],
                y_task,
                y_codec,
                y_codec_q,
            )
        else:
            losses = loss_fn(outputs["task_logits"], y_task)

        # Compute accuracies
        task_acc = compute_accuracy(outputs["task_logits"], y_task)

        # Accumulate metrics
        total_loss += losses["total_loss"].item()
        total_task_loss += losses["task_loss"].item()
        total_task_acc += task_acc
        num_batches += 1

        if method == "dann":
            total_codec_loss += losses["codec_loss"].item()
            total_codec_q_loss += losses["codec_q_loss"].item()
            codec_acc = compute_accuracy(outputs["codec_logits"], y_codec)
            codec_q_acc = compute_accuracy(outputs["codec_q_logits"], y_codec_q)
            total_codec_acc += codec_acc
            total_codec_q_acc += codec_q_acc

        # Collect scores for EER computation
        # Score convention: higher = more bonafide (class 0)
        # Use logit difference or softmax probability
        scores = torch.softmax(outputs["task_logits"], dim=-1)[:, 0]  # P(bonafide)
        all_scores.append(scores.cpu())
        all_labels.append(y_task.cpu())

        # Collect domain labels for breakdown
        if compute_domain_breakdown:
            all_codec_labels.append(y_codec.cpu())
            all_codec_q_labels.append(y_codec_q.cpu())

    # Compute EER
    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    from ..evaluation.metrics import compute_eer, compute_min_dcf

    eer, _ = compute_eer(all_scores, all_labels)
    min_dcf = compute_min_dcf(all_scores, all_labels)

    # Average metrics
    metrics = {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "task_acc": total_task_acc / num_batches,
        "eer": eer,
        "min_dcf": min_dcf,
    }

    if method == "dann":
        metrics["codec_loss"] = total_codec_loss / num_batches
        metrics["codec_q_loss"] = total_codec_q_loss / num_batches
        metrics["codec_acc"] = total_codec_acc / num_batches
        metrics["codec_q_acc"] = total_codec_q_acc / num_batches

    # Per-domain breakdown
    if compute_domain_breakdown and codec_vocab and codec_q_vocab:
        all_codec_labels = torch.cat(all_codec_labels).numpy()
        all_codec_q_labels = torch.cat(all_codec_q_labels).numpy()

        # Compute EER per CODEC
        codec_id_to_name = {v: k for k, v in codec_vocab.items()}
        metrics["per_codec"] = {}
        for codec_id in np.unique(all_codec_labels):
            mask = all_codec_labels == codec_id
            if mask.sum() > 10:  # Need enough samples
                codec_scores = all_scores[mask]
                codec_labels = all_labels[mask]
                if len(np.unique(codec_labels)) == 2:  # Need both classes
                    codec_eer, _ = compute_eer(codec_scores, codec_labels)
                    codec_name = codec_id_to_name.get(codec_id, str(codec_id))
                    metrics["per_codec"][codec_name] = {
                        "eer": float(codec_eer),
                        "n_samples": int(mask.sum()),
                    }

        # Compute EER per CODEC_Q
        codec_q_id_to_name = {v: k for k, v in codec_q_vocab.items()}
        metrics["per_codec_q"] = {}
        for codec_q_id in np.unique(all_codec_q_labels):
            mask = all_codec_q_labels == codec_q_id
            if mask.sum() > 10:
                cq_scores = all_scores[mask]
                cq_labels = all_labels[mask]
                if len(np.unique(cq_labels)) == 2:
                    cq_eer, _ = compute_eer(cq_scores, cq_labels)
                    cq_name = codec_q_id_to_name.get(codec_q_id, str(codec_q_id))
                    metrics["per_codec_q"][cq_name] = {
                        "eer": float(cq_eer),
                        "n_samples": int(mask.sum()),
                    }

    return metrics


def get_layer_weights(model: nn.Module) -> Optional[list[float]]:
    """Extract learned layer mixing weights from model.

    Args:
        model: Model with potential layer_pooling.

    Returns:
        List of layer weights or None if not available.
    """
    try:
        if hasattr(model, "backbone") and hasattr(model.backbone, "layer_pooling"):
            weights = model.backbone.layer_pooling.weights
            normalized = torch.softmax(weights, dim=0)
            return normalized.detach().cpu().tolist()
    except Exception:
        pass
    return None


class Trainer:
    """Training loop manager with checkpointing, early stopping, and comprehensive logging.

    Args:
        model: Model to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Optional LR scheduler.
        device: Device.
        run_dir: Directory for saving outputs.
        config: Full resolved config.
        method: Training method ('erm' or 'dann').
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        gradient_clip: Gradient clipping value.
        use_amp: Whether to use automatic mixed precision.
        log_interval: Logging interval (steps).
        val_interval: Validation interval (epochs).
        save_every_n_epochs: Checkpoint save interval.
        monitor_metric: Metric to monitor for best model.
        monitor_mode: 'min' or 'max'.
        lambda_scheduler: Optional DANN lambda scheduler.
        use_wandb: Whether to use wandb logging.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity (team or username).
        wandb_run_name: Wandb run name.
        wandb_tags: Optional list of wandb tags.
        batch_sample_rate: Rate of batches to log in detail (0.0-1.0).
        track_gradients: Whether to track gradient norms.
        log_domain_breakdown_every: Epochs between per-domain metric computation.
        codec_vocab: CODEC vocabulary for domain breakdown.
        codec_q_vocab: CODEC_Q vocabulary for domain breakdown.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        run_dir: Path,
        config: dict,
        method: str = "erm",
        max_epochs: int = 50,
        patience: int = 10,
        gradient_clip: float = 1.0,
        use_amp: bool = False,
        log_interval: int = 50,
        val_interval: int = 1,
        save_every_n_epochs: int = 5,
        monitor_metric: str = "eer",
        monitor_mode: str = "min",
        lambda_scheduler=None,
        use_wandb: bool = False,
        wandb_project: str = "asvspoof5-dann",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[list[str]] = None,
        batch_sample_rate: float = 0.02,
        track_gradients: bool = True,
        log_domain_breakdown_every: int = 5,
        codec_vocab: Optional[dict] = None,
        codec_q_vocab: Optional[dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = Path(run_dir)
        self.config = config
        self.method = method
        self.max_epochs = max_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.lambda_scheduler = lambda_scheduler
        self.batch_sample_rate = batch_sample_rate
        self.track_gradients = track_gradients
        self.log_domain_breakdown_every = log_domain_breakdown_every
        self.codec_vocab = codec_vocab
        self.codec_q_vocab = codec_q_vocab

        self.scaler = GradScaler("cuda") if use_amp else None

        # Setup directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize ExperimentLogger
        self.exp_logger = ExperimentLogger(
            run_dir=self.run_dir,
            run_name=wandb_run_name or self.run_dir.name,
            config=config,
            use_wandb=use_wandb and WANDB_AVAILABLE,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_tags=wandb_tags,
        )

        # Log experiment start
        self._log_experiment_start()

        # State
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.global_step = 0

        # Training log
        self.train_log = []

        # Save config
        save_config(config, self.run_dir / "config_resolved.yaml")

    def _log_experiment_start(self) -> None:
        """Log comprehensive experiment start event."""
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        self.exp_logger.log_model_summary(self.model, watch_gradients=False)

        # Dataset stats
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)

        self.exp_logger.log_dataset_stats(
            train_size=train_size,
            val_size=val_size,
        )

        # Log experiment start wide event
        backbone_name = self.config.get("backbone", {}).get("name", "unknown")
        start_event = {
            "method": self.method,
            "backbone": backbone_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.config.get("dataloader", {}).get("batch_size", 32),
            "learning_rate": self.config.get("training", {}).get("optimizer", {}).get("lr"),
            "model": {
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            "dataset": {
                "train_size": train_size,
                "val_size": val_size,
            },
        }

        if self.method == "dann":
            start_event["dann"] = {
                "lambda_init": self.config.get("dann", {}).get("lambda_", 0.1),
                "lambda_schedule": self.config.get("dann", {}).get("lambda_schedule", {}),
            }

        self.exp_logger.log_wide_event("experiment_start", start_event)

        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Run directory: {self.run_dir}")
        logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
        logger.info(f"Dataset: train={train_size}, val={val_size}")

    def train(self) -> dict:
        """Run full training loop.

        Returns:
            Dictionary of best metrics.
        """
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch

            # Update DANN lambda if scheduled
            current_lambda = None
            if self.lambda_scheduler is not None and self.method == "dann":
                current_lambda = self.lambda_scheduler.get_lambda(epoch)
                self.model.set_lambda(current_lambda)
                if hasattr(self.loss_fn, "set_lambda"):
                    self.loss_fn.set_lambda(current_lambda)
                logger.info(f"Epoch {epoch}: lambda = {current_lambda:.4f}")

            # Train
            train_metrics, self.global_step = train_epoch(
                self.model,
                self.train_loader,
                self.loss_fn,
                self.optimizer,
                self.device,
                scheduler=self.scheduler,
                scaler=self.scaler,
                gradient_clip=self.gradient_clip,
                method=self.method,
                log_interval=self.log_interval,
                batch_sample_rate=self.batch_sample_rate,
                track_gradients=self.track_gradients,
                exp_logger=self.exp_logger,
                global_step_start=self.global_step,
                log_step_interval=self.log_interval,
            )

            # Extract batch samples for separate logging
            batch_samples = train_metrics.pop("_batch_samples", [])

            # Log training metrics
            logger.info(
                f"Epoch {epoch} train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"task_acc={train_metrics['task_acc']:.4f}"
            )

            # Validate
            val_metrics = None
            if epoch % self.val_interval == 0:
                # Compute domain breakdown periodically
                compute_breakdown = (
                    epoch % self.log_domain_breakdown_every == 0
                    and self.codec_vocab is not None
                    and self.codec_q_vocab is not None
                )

                val_metrics = validate_epoch(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.device,
                    method=self.method,
                    compute_domain_breakdown=compute_breakdown,
                    codec_vocab=self.codec_vocab,
                    codec_q_vocab=self.codec_q_vocab,
                )

                logger.info(
                    f"Epoch {epoch} val: "
                    f"loss={val_metrics['loss']:.4f}, "
                    f"eer={val_metrics['eer']:.4f}, "
                    f"min_dcf={val_metrics['min_dcf']:.4f}"
                )

                # Check for improvement
                current_metric = val_metrics[self.monitor_metric]
                is_better = (
                    current_metric < self.best_metric
                    if self.monitor_mode == "min"
                    else current_metric > self.best_metric
                )

                if is_better:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0

                    # Save best model
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_metrics,
                        self.run_dir / "checkpoints" / "best.pt",
                        config=self.config,
                    )
                    logger.info(f"New best {self.monitor_metric}: {self.best_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1

            # Build epoch-complete wide event
            epoch_duration = time.time() - epoch_start_time
            epoch_event = {
                "epoch": epoch,
                "duration_sec": round(epoch_duration, 2),
                "train": {
                    k: v for k, v in train_metrics.items()
                    if isinstance(v, (int, float))
                },
                "is_best": val_metrics is not None and self.epochs_without_improvement == 0,
            }

            if val_metrics:
                epoch_event["val"] = {
                    k: v for k, v in val_metrics.items()
                    if isinstance(v, (int, float)) and not k.startswith("per_")
                }
                # Include domain breakdown if computed
                if "per_codec" in val_metrics:
                    epoch_event["per_codec"] = val_metrics["per_codec"]
                if "per_codec_q" in val_metrics:
                    epoch_event["per_codec_q"] = val_metrics["per_codec_q"]

            # Learning rate
            if self.scheduler is not None:
                epoch_event["learning_rate"] = self.scheduler.get_last_lr()[0]

            # DANN lambda
            if current_lambda is not None:
                epoch_event["lambda_domain"] = current_lambda

            # Layer weights
            layer_weights = get_layer_weights(self.model)
            if layer_weights:
                epoch_event["layer_weights"] = layer_weights

            # GPU memory
            gpu_mem = get_gpu_memory_usage()
            if gpu_mem is not None:
                epoch_event["gpu_memory_gb"] = gpu_mem

            # Log wide event
            self.exp_logger.log_wide_event("epoch_complete", epoch_event)

            # Record to log
            log_entry = {
                "epoch": epoch,
                "train": train_metrics,
            }
            if val_metrics:
                log_entry["val"] = {
                    k: v for k, v in val_metrics.items()
                    if not k.startswith("per_")
                }
            if current_lambda is not None:
                log_entry["lambda"] = current_lambda
            self.train_log.append(log_entry)

            # Periodic checkpoint
            if epoch % self.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    train_metrics,
                    self.run_dir / "checkpoints" / f"epoch_{epoch}.pt",
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            train_metrics,
            self.run_dir / "checkpoints" / "last.pt",
        )

        # Save training log
        with open(self.run_dir / "train_log.jsonl", "w") as f:
            for entry in self.train_log:
                f.write(json.dumps(entry, default=str) + "\n")

        # Build final metrics
        final_metrics = {
            "best_epoch": self.current_epoch - self.epochs_without_improvement,
            f"best_{self.monitor_metric}": self.best_metric,
            "final_epoch": self.current_epoch,
        }
        if self.train_log:
            final_metrics["final_val"] = self.train_log[-1].get("val", {})

        save_metrics(final_metrics, self.run_dir / "metrics_train.json")

        # Log training complete wide event
        self.exp_logger.log_wide_event("training_complete", final_metrics)

        # Set wandb summary
        self.exp_logger.set_summary({
            "best_epoch": final_metrics["best_epoch"],
            f"best_{self.monitor_metric}": self.best_metric,
            "total_epochs": self.current_epoch + 1,
        })

        # Finish logging
        self.exp_logger.finish()

        return final_metrics
