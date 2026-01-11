"""Training and validation loops."""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.io import save_checkpoint, save_config, save_metrics

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
) -> dict:
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
        log_interval: Logging interval.

    Returns:
        Dictionary of average metrics for the epoch.
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

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(pbar):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"].to(device)
        y_codec = batch["y_codec"].to(device)
        y_codec_q = batch["y_codec_q"].to(device)

        optimizer.zero_grad()

        # Forward pass
        use_amp = scaler is not None
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(device_type=device_type, enabled=use_amp):
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

        loss = losses["total_loss"]

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

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

        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "task_acc": f"{total_task_acc / num_batches:.4f}",
            })

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

    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    method: str = "erm",
) -> dict:
    """Validate for one epoch.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: Device.
        method: Training method ('erm' or 'dann').

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

    return metrics


class Trainer:
    """Training loop manager with checkpointing and early stopping.

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

        self.scaler = GradScaler("cuda") if use_amp else None

        # Wandb setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("Wandb requested but not installed. Skipping.")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    config=config,
                    dir=str(run_dir),
                )
                logger.info(f"Wandb initialized: {wandb.run.url}")

        # State
        self.best_metric = float("inf") if monitor_mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.global_step = 0

        # Logging
        self.train_log = []

        # Setup directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Save config
        save_config(config, self.run_dir / "config_resolved.yaml")

    def train(self) -> dict:
        """Run full training loop.

        Returns:
            Dictionary of best metrics.
        """
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Run directory: {self.run_dir}")

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Update DANN lambda if scheduled
            if self.lambda_scheduler is not None and self.method == "dann":
                new_lambda = self.lambda_scheduler.get_lambda(epoch)
                self.model.set_lambda(new_lambda)
                if hasattr(self.loss_fn, "set_lambda"):
                    self.loss_fn.set_lambda(new_lambda)
                logger.info(f"Epoch {epoch}: lambda = {new_lambda:.4f}")

            # Train
            train_metrics = train_epoch(
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
            )

            # Log training metrics
            logger.info(
                f"Epoch {epoch} train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"task_acc={train_metrics['task_acc']:.4f}"
            )

            # Wandb logging - training
            if self.use_wandb:
                wandb_train = {f"train/{k}": v for k, v in train_metrics.items()}
                wandb_train["epoch"] = epoch
                if self.scheduler is not None:
                    wandb_train["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(wandb_train, step=epoch)

            # Validate
            if epoch % self.val_interval == 0:
                val_metrics = validate_epoch(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.device,
                    method=self.method,
                )

                logger.info(
                    f"Epoch {epoch} val: "
                    f"loss={val_metrics['loss']:.4f}, "
                    f"eer={val_metrics['eer']:.4f}, "
                    f"min_dcf={val_metrics['min_dcf']:.4f}"
                )

                # Wandb logging - validation
                if self.use_wandb:
                    wandb_val = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb.log(wandb_val, step=epoch)

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

                # Record to log
                log_entry = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
                if self.lambda_scheduler is not None and self.method == "dann":
                    log_entry["lambda"] = self.lambda_scheduler.get_lambda(epoch)
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
                f.write(json.dumps(entry) + "\n")

        # Save final metrics
        final_metrics = {
            "best_epoch": self.current_epoch - self.epochs_without_improvement,
            f"best_{self.monitor_metric}": self.best_metric,
            "final_epoch": self.current_epoch,
        }
        if self.train_log:
            final_metrics["final_val"] = self.train_log[-1].get("val", {})

        save_metrics(final_metrics, self.run_dir / "metrics_train.json")

        # Wandb final logging
        if self.use_wandb:
            wandb.log(final_metrics)
            wandb.finish()

        return final_metrics
