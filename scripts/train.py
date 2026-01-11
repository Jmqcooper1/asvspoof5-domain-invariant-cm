#!/usr/bin/env python3
"""Training entrypoint for ERM and DANN models.

Usage:
    # ERM with WavLM
    python scripts/train.py --config configs/wavlm_erm.yaml

    # DANN with WavLM
    python scripts/train.py --config configs/wavlm_dann.yaml

    # With separate config files
    python scripts/train.py \
        --train-config configs/train/erm.yaml \
        --model-config configs/model/wavlm_base.yaml \
        --data-config configs/data/asvspoof5_track1.yaml

    # Override run name
    python scripts/train.py --config configs/wavlm_dann.yaml --name my_experiment
"""

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import torch

from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset,
    AudioCollator,
    load_vocab,
)
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)
from asvspoof5_domain_invariant_cm.training import (
    LambdaScheduler,
    Trainer,
    build_loss,
    build_lr_scheduler,
    build_optimizer,
)
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_manifest_path,
    get_manifests_dir,
    get_run_dir,
    load_config,
    merge_configs,
    set_seed,
)

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
        default=None,
        help="Path to combined config (contains all settings)",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Path to training config (erm.yaml or dann.yaml)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to model config (wavlm_base.yaml, w2v2_base.yaml, etc.)",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/asvspoof5_track1.yaml"),
        help="Path to data config",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    return parser.parse_args()


def load_configs(args) -> dict:
    """Load and merge configuration files."""
    if args.config is not None:
        # Single combined config
        config = load_config(args.config)
    else:
        # Separate config files
        configs = []

        if args.data_config and args.data_config.exists():
            configs.append(load_config(args.data_config))

        if args.model_config and args.model_config.exists():
            configs.append(load_config(args.model_config))

        if args.train_config and args.train_config.exists():
            configs.append(load_config(args.train_config))

        if not configs:
            raise ValueError(
                "No config files provided. Use --config or --train-config + --model-config"
            )

        config = merge_configs(*configs)

    # Override with CLI args
    if args.seed is not None:
        config["seed"] = args.seed

    return config


def build_model(config: dict, num_codecs: int, num_codec_qs: int) -> torch.nn.Module:
    """Build model from config."""
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    # Backbone
    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=backbone_cfg.get("freeze", True),
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    # Pooling
    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    # Calculate projection input dim
    if pooling_method == "stats":
        proj_input_dim = backbone.hidden_size * 2
    else:
        proj_input_dim = backbone.hidden_size

    # Projection head
    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)

    # Task head
    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

    # Build model based on method
    method = training_cfg.get("method", "erm")

    if method == "dann":
        dann_cfg = config.get("dann", {})
        disc_cfg = dann_cfg.get("discriminator", {})

        domain_discriminator = MultiHeadDomainDiscriminator(
            input_dim=repr_dim,
            num_codecs=num_codecs,
            num_codec_qs=num_codec_qs,
            hidden_dim=disc_cfg.get("hidden_dim", 256),
            dropout=disc_cfg.get("dropout", 0.1),
        )

        model = DANNModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
            domain_discriminator=domain_discriminator,
            lambda_=dann_cfg.get("lambda_", 0.1),
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    return model


def main():
    args = parse_args()

    # Load config
    config = load_configs(args)

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed, deterministic=config.get("deterministic", True))

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Setup run directory
    if args.name:
        run_name = args.name
    else:
        method = config.get("training", {}).get("method", "erm")
        backbone_name = config.get("backbone", {}).get("name", "wavlm")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{backbone_name}_{method}_{timestamp}"

    run_dir = get_run_dir(run_name)
    logger.info(f"Run directory: {run_dir}")

    # Load vocabularies
    manifests_dir = get_manifests_dir()
    codec_vocab = load_vocab(manifests_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(manifests_dir / "codec_q_vocab.json")

    logger.info(f"CODEC classes: {len(codec_vocab)}")
    logger.info(f"CODEC_Q classes: {len(codec_q_vocab)}")

    # Copy vocabs to run dir
    shutil.copy(manifests_dir / "codec_vocab.json", run_dir / "codec_vocab.json")
    shutil.copy(manifests_dir / "codec_q_vocab.json", run_dir / "codec_q_vocab.json")

    # Data config
    data_cfg = config.get("dataset", config.get("data", {}))
    audio_cfg = config.get("audio", {})
    dataloader_cfg = config.get("dataloader", {})

    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)
    batch_size = dataloader_cfg.get("batch_size", 32)
    num_workers = dataloader_cfg.get("num_workers", 4)

    # Create datasets
    train_dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path("train"),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="train",
    )

    val_dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path("dev"),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    fixed_length = int(max_duration * sample_rate)

    train_collator = AudioCollator(fixed_length=fixed_length, mode="train")
    val_collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collator,
        drop_last=False,
        pin_memory=True,
    )

    # Build model
    model = build_model(config, len(codec_vocab), len(codec_q_vocab))
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Training config
    training_cfg = config.get("training", {})
    method = training_cfg.get("method", "erm")
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})

    # Build optimizer
    optimizer = build_optimizer(
        model,
        name=optimizer_cfg.get("name", "adamw"),
        lr=optimizer_cfg.get("lr", 1e-4),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
    )

    # Build scheduler
    num_training_steps = len(train_loader) * training_cfg.get("max_epochs", 50)
    scheduler = build_lr_scheduler(
        optimizer,
        name=scheduler_cfg.get("name", "cosine"),
        num_warmup_steps=scheduler_cfg.get("warmup_steps", 500),
        num_training_steps=num_training_steps,
        min_lr_ratio=scheduler_cfg.get("min_lr", 1e-6) / optimizer_cfg.get("lr", 1e-4),
    )

    # Build loss
    loss_cfg = config.get("loss", {})
    task_loss_cfg = loss_cfg.get("task", {})
    dann_cfg = config.get("dann", {})

    loss_fn = build_loss(
        method=method,
        task_label_smoothing=task_loss_cfg.get("label_smoothing", 0.0),
        lambda_domain=dann_cfg.get("lambda_", 0.1) if method == "dann" else 0.0,
    )

    # Build lambda scheduler (for DANN)
    lambda_scheduler = None
    if method == "dann":
        lambda_sched_cfg = dann_cfg.get("lambda_schedule", {})
        if lambda_sched_cfg.get("enabled", False):
            lambda_scheduler = LambdaScheduler(
                schedule_type=lambda_sched_cfg.get("type", "linear"),
                start_value=lambda_sched_cfg.get("start", 0.0),
                end_value=lambda_sched_cfg.get("end", 1.0),
                warmup_epochs=lambda_sched_cfg.get("warmup_epochs", 5),
                total_epochs=training_cfg.get("max_epochs", 50),
            )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        run_dir=run_dir,
        config=config,
        method=method,
        max_epochs=training_cfg.get("max_epochs", 50),
        patience=training_cfg.get("patience", 10),
        gradient_clip=training_cfg.get("gradient_clip", 1.0),
        use_amp=args.amp,
        log_interval=config.get("logging", {}).get("log_every_n_steps", 50),
        val_interval=config.get("logging", {}).get("val_every_n_epochs", 1),
        save_every_n_epochs=training_cfg.get("save_every_n_epochs", 5),
        monitor_metric=training_cfg.get("monitor_metric", "eer"),
        monitor_mode=training_cfg.get("monitor_mode", "min"),
        lambda_scheduler=lambda_scheduler,
    )

    # Train
    logger.info("=" * 60)
    logger.info(f"Training {method.upper()} model")
    logger.info("=" * 60)

    final_metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best {training_cfg.get('monitor_metric', 'eer')}: {final_metrics.get('best_eer', 'N/A')}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
