#!/usr/bin/env python3
"""Held-out codec domain generalization experiment.

This script evaluates domain generalization by training models with one synthetic CODEC
held out, then testing on the held-out CODEC to measure the generalization gap.

Compares ERM vs DANN to show whether domain-adversarial training improves
generalization to unseen codecs.

Usage:
    python scripts/run_held_out_codec.py --config configs/wavlm_dann.yaml
    python scripts/run_held_out_codec.py --top-n 3 --output-dir runs/held_out
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.data import (
    ASVspoof5Dataset, 
    load_vocab, 
    create_augmentor,
    SYNTHETIC_CODEC_VOCAB,
    SYNTHETIC_QUALITY_VOCAB,
)
from asvspoof5_domain_invariant_cm.data.audio import AudioCollator
from asvspoof5_domain_invariant_cm.evaluation import (
    compute_domain_gap,
    compute_eer,
    compute_min_dcf,
)
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_manifest_path,
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
    parser = argparse.ArgumentParser(
        description="Held-out codec domain generalization experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/wavlm_dann.yaml"),
        help="Base config file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top synthetic CODECs to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing checkpoints",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-held-out",
        help="Wandb project name",
    )
    return parser.parse_args()


def get_synthetic_codecs(config: dict) -> list[str]:
    """Get codec families from augmentation config.

    Args:
        config: Configuration dictionary.

    Returns:
        List of synthetic codec names.
    """
    aug_config = config.get("augmentation", {})
    if not aug_config.get("enabled", False):
        raise ValueError("Augmentation must be enabled for held-out codec experiment")
    
    # Get supported codecs from augmentation config
    codecs = aug_config.get("codecs", ["MP3", "AAC", "OPUS"])
    return codecs


def create_augmentor_excluding_codec(
    config: dict, 
    excluded_codec: str,
    sample_rate: int = 16000,
) -> object:
    """Create augmentor that excludes a specific codec.

    Args:
        config: Configuration dictionary.
        excluded_codec: Codec to exclude from augmentation.
        sample_rate: Sample rate for audio.

    Returns:
        Augmentor with excluded codec removed.
    """
    aug_config = config.get("augmentation", {}).copy()
    
    # Remove the held-out codec from the supported codecs list
    original_codecs = aug_config.get("codecs", ["MP3", "AAC", "OPUS"])
    remaining_codecs = [c for c in original_codecs if c != excluded_codec]
    
    if len(remaining_codecs) == 0:
        raise ValueError(f"Cannot exclude {excluded_codec}: no remaining codecs")
    
    aug_config["codecs"] = remaining_codecs
    aug_config["sample_rate"] = sample_rate
    
    logger.info(f"Creating augmentor with codecs: {remaining_codecs} (excluding {excluded_codec})")
    
    return create_augmentor(aug_config)


def create_augmentor_only_codec(
    config: dict, 
    only_codec: str,
    sample_rate: int = 16000,
) -> object:
    """Create augmentor that uses only a specific codec.

    Args:
        config: Configuration dictionary.
        only_codec: Codec to use exclusively.
        sample_rate: Sample rate for audio.

    Returns:
        Augmentor with only the specified codec.
    """
    aug_config = config.get("augmentation", {}).copy()
    aug_config["codecs"] = [only_codec]
    aug_config["sample_rate"] = sample_rate
    # Force codec application
    aug_config["codec_prob"] = 1.0
    
    return create_augmentor(aug_config)


def evaluate_model_on_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    augmentor: object,
    device: torch.device,
    batch_size: int = 32,
    max_duration_sec: float = 6.0,
    sample_rate: int = 16000,
) -> dict:
    """Evaluate a model on a given dataframe split with augmentation.

    Args:
        model: Trained model.
        df: DataFrame with samples to evaluate.
        augmentor: Augmentor to apply (or None).
        device: Device to use.
        batch_size: Batch size.
        max_duration_sec: Max audio duration.
        sample_rate: Sample rate.

    Returns:
        Dictionary with EER and minDCF metrics.
    """
    from torch.utils.data import DataLoader
    import tempfile

    # Save temporary manifest
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = Path(f.name)
        df.to_parquet(temp_path)

    # Use synthetic vocab if augmentation is enabled
    if augmentor is not None:
        codec_vocab = augmentor.codec_vocab
        codec_q_vocab = SYNTHETIC_QUALITY_VOCAB
        use_synthetic_labels = True
    else:
        # Fallback to original vocabs
        from asvspoof5_domain_invariant_cm.utils import get_manifests_dir
        manifests_dir = get_manifests_dir()
        codec_vocab = load_vocab(manifests_dir / "codec_vocab.json")
        codec_q_vocab = load_vocab(manifests_dir / "codec_q_vocab.json")
        use_synthetic_labels = False

    dataset = ASVspoof5Dataset(
        manifest_path=temp_path,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration_sec,
        sample_rate=sample_rate,
        mode="eval",
        augmentor=augmentor,
        use_synthetic_labels=use_synthetic_labels,
    )

    fixed_length = int(max_duration_sec * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
    )

    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)
            y_task = batch["y_task"]

            outputs = model(waveform, attention_mask, lengths)

            # Score convention: higher = more bonafide (class 0)
            scores = torch.softmax(outputs["task_logits"], dim=-1)[:, 0]
            all_scores.append(scores.cpu())
            all_labels.append(y_task)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    eer, eer_threshold = compute_eer(all_scores, all_labels)
    min_dcf = compute_min_dcf(all_scores, all_labels)

    # Cleanup temp file
    temp_path.unlink()

    return {
        "eer": float(eer),
        "min_dcf": float(min_dcf),
        "n_samples": len(all_scores),
        "n_bonafide": int(np.sum(all_labels == 0)),
        "n_spoof": int(np.sum(all_labels == 1)),
    }


def run_held_out_experiment(
    held_out_codec: str,
    config: dict,
    output_dir: Path,
    seed: int,
    skip_training: bool = False,
    use_wandb: bool = False,
    wandb_project: str = "asvspoof5-held-out",
) -> dict:
    """Run a single held-out codec experiment.

    Args:
        held_out_codec: CODEC to hold out.
        config: Full config dictionary.
        output_dir: Output directory for this experiment.
        seed: Random seed.
        skip_training: If True, skip training and load existing checkpoints.
        use_wandb: Whether to enable wandb logging.
        wandb_project: Wandb project name.

    Returns:
        Dictionary with results for this held-out codec.
    """
    from asvspoof5_domain_invariant_cm.models import DANNModel, ERMModel, create_backbone
    from asvspoof5_domain_invariant_cm.models.heads import ClassifierHead, ProjectionHead, create_pooling
    from asvspoof5_domain_invariant_cm.training import Trainer, build_loss, build_lr_scheduler, build_optimizer

    set_seed(seed)
    device = get_device()

    # Create output directory
    exp_dir = output_dir / f"held_out_{held_out_codec}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load train/dev manifests
    train_manifest = get_manifest_path("train")
    dev_manifest = get_manifest_path("dev")
    train_df = pd.read_parquet(train_manifest)
    dev_df = pd.read_parquet(dev_manifest)

    results = {
        "held_out_codec": held_out_codec,
        "train_samples": len(train_df),
        "dev_samples": len(dev_df),
    }

    # Train and evaluate both ERM and DANN
    for method in ["erm", "dann"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {method.upper()} with held-out CODEC: {held_out_codec}")
        logger.info(f"{'='*60}")

        method_dir = exp_dir / method
        checkpoint_path = method_dir / "checkpoints" / "best.pt"

        if skip_training and checkpoint_path.exists():
            logger.info(f"Loading existing checkpoint: {checkpoint_path}")
        else:
            # Create augmentor excluding the held-out codec (for training)
            train_augmentor = create_augmentor_excluding_codec(
                config, held_out_codec, sample_rate=config["audio"]["sample_rate"]
            )

            # Create datasets for training
            train_dataset = ASVspoof5Dataset(
                manifest_path=train_manifest,
                codec_vocab=train_augmentor.codec_vocab,
                codec_q_vocab=SYNTHETIC_QUALITY_VOCAB,
                max_duration_sec=config["audio"]["max_duration_sec"],
                sample_rate=config["audio"]["sample_rate"],
                mode="train",
                augmentor=train_augmentor,
                use_synthetic_labels=True,
            )

            val_dataset = ASVspoof5Dataset(
                manifest_path=dev_manifest,
                codec_vocab=train_augmentor.codec_vocab,
                codec_q_vocab=SYNTHETIC_QUALITY_VOCAB,
                max_duration_sec=config["audio"]["max_duration_sec"],
                sample_rate=config["audio"]["sample_rate"],
                mode="eval",
                augmentor=train_augmentor,
                use_synthetic_labels=True,
            )

            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")

            # Create dataloaders
            fixed_length = int(config["audio"]["max_duration_sec"] * config["audio"]["sample_rate"])
            train_collator = AudioCollator(fixed_length=fixed_length, mode="train")
            val_collator = AudioCollator(fixed_length=fixed_length, mode="eval")

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config["dataloader"]["batch_size"],
                shuffle=True,
                num_workers=config["dataloader"].get("num_workers", 4),
                collate_fn=train_collator,
                drop_last=True,
                pin_memory=True,
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config["dataloader"]["batch_size"],
                shuffle=False,
                num_workers=config["dataloader"].get("num_workers", 4),
                collate_fn=val_collator,
                drop_last=False,
                pin_memory=True,
            )

            # Build model
            backbone_config = config["backbone"]
            backbone = create_backbone(
                backbone_config["name"],
                freeze=backbone_config.get("freeze", False),
                layer_selection=backbone_config.get("layer_selection", "last_k"),
                layer_selection_k=backbone_config.get("layer_selection_k", 4),
            )

            pooling = create_pooling(
                config["pooling"]["method"],
                input_dim=backbone.hidden_size,
            )

            pooling_output_dim = (
                backbone.hidden_size * 2
                if config["pooling"]["method"] == "stats"
                else backbone.hidden_size
            )

            projection = ProjectionHead(
                input_dim=pooling_output_dim,
                hidden_dim=config["projection"]["hidden_dim"],
                output_dim=config["projection"]["output_dim"],
                dropout=config["projection"].get("dropout", 0.1),
            )

            task_head = ClassifierHead(
                input_dim=config["projection"]["output_dim"],
                num_classes=2,
            )

            if method == "dann":
                from asvspoof5_domain_invariant_cm.models import (
                    GradientReversalLayer,
                    MultiHeadDomainDiscriminator,
                )

                grl = GradientReversalLayer(lambda_=config["dann"].get("lambda_", 0.0))
                domain_discriminator = MultiHeadDomainDiscriminator(
                    input_dim=config["projection"]["output_dim"],
                    hidden_dim=config["dann"]["discriminator"]["hidden_dim"],
                    num_codecs=len(train_augmentor.codec_vocab),
                    num_codec_qs=len(SYNTHETIC_QUALITY_VOCAB),
                    dropout=config["dann"]["discriminator"].get("dropout", 0.1),
                )

                model = DANNModel(
                    backbone=backbone,
                    pooling=pooling,
                    projection=projection,
                    task_head=task_head,
                    grl=grl,
                    domain_discriminator=domain_discriminator,
                )
            else:
                model = ERMModel(
                    backbone=backbone,
                    pooling=pooling,
                    projection=projection,
                    task_head=task_head,
                )

            model = model.to(device)

            # Build optimizer and scheduler
            optimizer = build_optimizer(model, config["training"]["optimizer"])
            total_steps = len(train_loader) * config["training"]["max_epochs"]
            scheduler = build_lr_scheduler(
                optimizer, config["training"]["scheduler"], total_steps
            )

            # Build loss
            loss_fn = build_loss(
                method=method,
                config=config["training"],
                num_codecs=len(train_augmentor.codec_vocab),
                num_codec_qs=len(SYNTHETIC_QUALITY_VOCAB),
            )

            # Lambda scheduler for DANN
            lambda_scheduler = None
            if method == "dann":
                from asvspoof5_domain_invariant_cm.training.sched import LambdaScheduler

                lambda_sched_cfg = config["dann"].get("lambda_schedule", {})
                lambda_scheduler = LambdaScheduler(
                    total_epochs=config["training"]["max_epochs"],
                    schedule_type=lambda_sched_cfg.get("type", "linear"),
                    start_value=lambda_sched_cfg.get("start", 0.0),
                    end_value=lambda_sched_cfg.get("end", 1.0),
                    warmup_epochs=lambda_sched_cfg.get("warmup_epochs", 0),
                )

            # Save vocabs to run directory
            import json
            with open(method_dir / "codec_vocab.json", "w") as f:
                json.dump(train_augmentor.codec_vocab, f, indent=2)
            with open(method_dir / "codec_q_vocab.json", "w") as f:
                json.dump(SYNTHETIC_QUALITY_VOCAB, f, indent=2)

            # Train
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                run_dir=method_dir,
                config=config,
                method=method,
                max_epochs=config["training"]["max_epochs"],
                patience=config["training"].get("patience", 10),
                min_delta=config["training"].get("min_delta", 0.001),
                train_loss_threshold=config["training"].get("train_loss_threshold", 0.01),
                plateau_patience=config["training"].get("plateau_patience", 3),
                gradient_clip=config["training"].get("gradient_clip", 1.0),
                use_amp=config["training"].get("use_amp", False),
                lambda_scheduler=lambda_scheduler,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_run_name=f"held_out_{held_out_codec}_{method}",
                wandb_tags=["held-out", held_out_codec, method],
                codec_vocab=train_augmentor.codec_vocab,
                codec_q_vocab=SYNTHETIC_QUALITY_VOCAB,
            )

            trainer.train()

        # Load best model and evaluate
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Rebuild model for loading (using the saved vocab)
        saved_codec_vocab = load_vocab(method_dir / "codec_vocab.json")
        saved_codec_q_vocab = load_vocab(method_dir / "codec_q_vocab.json")

        backbone_config = config["backbone"]
        backbone = create_backbone(
            backbone_config["name"],
            freeze=backbone_config.get("freeze", False),
            layer_selection=backbone_config.get("layer_selection", "last_k"),
            layer_selection_k=backbone_config.get("layer_selection_k", 4),
        )

        pooling = create_pooling(
            config["pooling"]["method"],
            input_dim=backbone.hidden_size,
        )

        pooling_output_dim = (
            backbone.hidden_size * 2
            if config["pooling"]["method"] == "stats"
            else backbone.hidden_size
        )

        projection = ProjectionHead(
            input_dim=pooling_output_dim,
            hidden_dim=config["projection"]["hidden_dim"],
            output_dim=config["projection"]["output_dim"],
            dropout=config["projection"].get("dropout", 0.1),
        )

        task_head = ClassifierHead(
            input_dim=config["projection"]["output_dim"],
            num_classes=2,
        )

        if method == "dann":
            from asvspoof5_domain_invariant_cm.models import (
                GradientReversalLayer,
                MultiHeadDomainDiscriminator,
            )

            grl = GradientReversalLayer()
            domain_discriminator = MultiHeadDomainDiscriminator(
                input_dim=config["projection"]["output_dim"],
                hidden_dim=config["dann"]["discriminator"]["hidden_dim"],
                num_codecs=len(saved_codec_vocab),
                num_codec_qs=len(saved_codec_q_vocab),
            )

            model = DANNModel(
                backbone=backbone,
                pooling=pooling,
                projection=projection,
                task_head=task_head,
                grl=grl,
                domain_discriminator=domain_discriminator,
            )
        else:
            model = ERMModel(
                backbone=backbone,
                pooling=pooling,
                projection=projection,
                task_head=task_head,
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        # Create evaluation augmentors
        # In-domain: use the training augmentor (excludes held-out codec)
        in_domain_augmentor = create_augmentor_excluding_codec(
            config, held_out_codec, sample_rate=config["audio"]["sample_rate"]
        )
        
        # Out-of-domain: use only the held-out codec
        out_domain_augmentor = create_augmentor_only_codec(
            config, held_out_codec, sample_rate=config["audio"]["sample_rate"]
        )

        # Evaluate on in-domain (dev set with remaining codecs)
        in_domain_metrics = evaluate_model_on_split(
            model, dev_df, in_domain_augmentor, device,
            batch_size=config["dataloader"]["batch_size"],
            max_duration_sec=config["audio"]["max_duration_sec"],
            sample_rate=config["audio"]["sample_rate"],
        )

        # Evaluate on out-of-domain (dev set with only held-out codec)
        out_domain_metrics = evaluate_model_on_split(
            model, dev_df, out_domain_augmentor, device,
            batch_size=config["dataloader"]["batch_size"],
            max_duration_sec=config["audio"]["max_duration_sec"],
            sample_rate=config["audio"]["sample_rate"],
        )

        # Compute domain gap
        gap = compute_domain_gap(in_domain_metrics, out_domain_metrics)

        results[f"{method}_in_domain"] = in_domain_metrics
        results[f"{method}_out_domain"] = out_domain_metrics
        results[f"{method}_gap"] = gap

        logger.info(f"{method.upper()} Results:")
        logger.info(f"  In-domain EER: {in_domain_metrics['eer']:.4f}")
        logger.info(f"  Out-domain EER: {out_domain_metrics['eer']:.4f}")
        logger.info(f"  EER Gap: {gap['eer_gap']:.4f}")

    return results


def main():
    args = parse_args()

    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Ensure augmentation is enabled
    if not config.get("augmentation", {}).get("enabled", False):
        raise ValueError("Augmentation must be enabled in config for held-out codec experiment")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from asvspoof5_domain_invariant_cm.utils import get_runs_dir
        output_dir = get_runs_dir() / "held_out_codec"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Optional wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=f"held_out_top{args.top_n}",
                config=config,
            )
        except ImportError:
            logger.warning("Wandb not installed, skipping logging")

    # Get top N synthetic codecs
    all_codecs = get_synthetic_codecs(config)
    top_codecs = all_codecs[:args.top_n]
    logger.info(f"Top {args.top_n} synthetic CODECs: {top_codecs}")

    # Run experiments
    all_results = []
    for codec in top_codecs:
        result = run_held_out_experiment(
            held_out_codec=codec,
            config=config,
            output_dir=output_dir,
            seed=args.seed,
            skip_training=args.skip_training,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
        )
        all_results.append(result)

    # Create summary table
    summary_rows = []
    for r in all_results:
        row = {
            "held_out_codec": r["held_out_codec"],
            "train_samples": r["train_samples"],
            "dev_samples": r["dev_samples"],
        }

        for method in ["erm", "dann"]:
            in_key = f"{method}_in_domain"
            out_key = f"{method}_out_domain"
            gap_key = f"{method}_gap"

            if in_key in r:
                row[f"{method}_in_eer"] = r[in_key]["eer"]
                row[f"{method}_in_dcf"] = r[in_key]["min_dcf"]

            if out_key in r:
                row[f"{method}_out_eer"] = r[out_key]["eer"]
                row[f"{method}_out_dcf"] = r[out_key]["min_dcf"]

            if gap_key in r:
                row[f"{method}_eer_gap"] = r[gap_key]["eer_gap"]
                row[f"{method}_dcf_gap"] = r[gap_key]["min_dcf_gap"]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save results
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    logger.info(f"\nSummary saved to: {output_dir / 'summary.csv'}")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("HELD-OUT CODEC EXPERIMENT SUMMARY")
    logger.info("=" * 80)
    logger.info(summary_df.to_string(index=False))

    # Compute average gaps
    if "erm_eer_gap" in summary_df.columns and "dann_eer_gap" in summary_df.columns:
        avg_erm_gap = summary_df["erm_eer_gap"].mean()
        avg_dann_gap = summary_df["dann_eer_gap"].mean()

        logger.info(f"\nAverage EER Gap:")
        logger.info(f"  ERM:  {avg_erm_gap:.4f}")
        logger.info(f"  DANN: {avg_dann_gap:.4f}")
        logger.info(f"  Improvement: {avg_erm_gap - avg_dann_gap:.4f}")

        if args.wandb:
            try:
                import wandb
                wandb.log({
                    "avg_erm_eer_gap": avg_erm_gap,
                    "avg_dann_eer_gap": avg_dann_gap,
                    "improvement": avg_erm_gap - avg_dann_gap,
                })
                wandb.finish()
            except Exception:
                pass

    logger.info("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())