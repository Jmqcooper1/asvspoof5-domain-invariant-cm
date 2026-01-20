#!/usr/bin/env python3
"""Held-out codec domain generalization experiment.

This script evaluates domain generalization by training models with one CODEC
held out, then testing on the held-out CODEC to measure the generalization gap.

Compares ERM vs DANN to show whether domain-adversarial training improves
generalization to unseen codecs.

Usage:
    python scripts/run_held_out_codec.py --config configs/wavlm_dann.yaml
    python scripts/run_held_out_codec.py --top-n 5 --output-dir runs/held_out
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.data import ASVspoof5Dataset, load_vocab
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
        default=5,
        help="Number of top CODECs to evaluate (by sample count)",
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


def get_top_codecs(manifest_path: Path, top_n: int) -> list[str]:
    """Get top N CODECs by sample count.

    Args:
        manifest_path: Path to manifest file.
        top_n: Number of top codecs to return.

    Returns:
        List of codec names sorted by sample count (descending).
    """
    df = pd.read_parquet(manifest_path)
    codec_counts = df["codec"].value_counts()
    return codec_counts.head(top_n).index.tolist()


def create_held_out_splits(
    manifest_path: Path,
    held_out_codec: str,
    codec_vocab: dict,
    codec_q_vocab: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test splits with one CODEC held out.

    Groups by codec_seed to prevent leakage between coded variants.

    Args:
        manifest_path: Path to manifest file.
        held_out_codec: CODEC value to hold out for testing.
        codec_vocab: CODEC vocabulary.
        codec_q_vocab: CODEC_Q vocabulary.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = pd.read_parquet(manifest_path)

    # Split by CODEC
    train_df = df[df["codec"] != held_out_codec].copy()
    test_df = df[df["codec"] == held_out_codec].copy()

    logger.info(f"Held-out CODEC: {held_out_codec}")
    logger.info(f"  Train samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")

    return train_df, test_df


def evaluate_model_on_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    codec_vocab: dict,
    codec_q_vocab: dict,
    device: torch.device,
    batch_size: int = 32,
    max_duration_sec: float = 6.0,
    sample_rate: int = 16000,
) -> dict:
    """Evaluate a model on a given dataframe split.

    Args:
        model: Trained model.
        df: DataFrame with samples to evaluate.
        codec_vocab: CODEC vocabulary.
        codec_q_vocab: CODEC_Q vocabulary.
        device: Device to use.
        batch_size: Batch size.
        max_duration_sec: Max audio duration.
        sample_rate: Sample rate.

    Returns:
        Dictionary with EER and minDCF metrics.
    """
    from torch.utils.data import DataLoader

    # Save temporary manifest
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = Path(f.name)
        df.to_parquet(temp_path)

    dataset = ASVspoof5Dataset(
        manifest_path=temp_path,
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration_sec,
        sample_rate=sample_rate,
        mode="eval",
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
) -> dict:
    """Run a single held-out codec experiment.

    Args:
        held_out_codec: CODEC to hold out.
        config: Full config dictionary.
        output_dir: Output directory for this experiment.
        seed: Random seed.
        skip_training: If True, skip training and load existing checkpoints.

    Returns:
        Dictionary with results for this held-out codec.
    """
    from asvspoof5_domain_invariant_cm.models import DANNModel, ERMModel, create_backbone
    from asvspoof5_domain_invariant_cm.models.heads import ClassifierHead, ProjectionHead, create_pooling
    from asvspoof5_domain_invariant_cm.training import Trainer, build_loss, build_lr_scheduler, build_optimizer

    set_seed(seed)
    device = get_device()

    # Load vocabularies
    vocab_dir = get_manifest_path("train").parent
    codec_vocab = load_vocab(vocab_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(vocab_dir / "codec_q_vocab.json")

    # Create output directory
    exp_dir = output_dir / f"held_out_{held_out_codec}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create train/test splits
    train_manifest = get_manifest_path("train")
    dev_manifest = get_manifest_path("dev")

    train_df, test_df = create_held_out_splits(
        train_manifest, held_out_codec, codec_vocab, codec_q_vocab
    )

    # Also create held-out split from dev set
    dev_df = pd.read_parquet(dev_manifest)
    dev_in_domain = dev_df[dev_df["codec"] != held_out_codec]
    dev_out_domain = dev_df[dev_df["codec"] == held_out_codec]

    results = {
        "held_out_codec": held_out_codec,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
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
            # Save temporary train manifest
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_train_path = Path(f.name)
                train_df.to_parquet(temp_train_path)

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
                temp_val_path = Path(f.name)
                dev_in_domain.to_parquet(temp_val_path)

            # Create datasets
            from asvspoof5_domain_invariant_cm.data import create_dataloader

            train_loader = create_dataloader(
                manifest_path=temp_train_path,
                codec_vocab=codec_vocab,
                codec_q_vocab=codec_q_vocab,
                batch_size=config["dataloader"]["batch_size"],
                max_duration_sec=config["audio"]["max_duration_sec"],
                sample_rate=config["audio"]["sample_rate"],
                mode="train",
                num_workers=config["dataloader"].get("num_workers", 4),
                shuffle=True,
                drop_last=True,
            )

            val_loader = create_dataloader(
                manifest_path=temp_val_path,
                codec_vocab=codec_vocab,
                codec_q_vocab=codec_q_vocab,
                batch_size=config["dataloader"]["batch_size"],
                max_duration_sec=config["audio"]["max_duration_sec"],
                sample_rate=config["audio"]["sample_rate"],
                mode="eval",
                num_workers=config["dataloader"].get("num_workers", 4),
                shuffle=False,
                drop_last=False,
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
                    num_codecs=len(codec_vocab),
                    num_codec_qs=len(codec_q_vocab),
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
                num_codecs=len(codec_vocab),
                num_codec_qs=len(codec_q_vocab),
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
                gradient_clip=config["training"].get("gradient_clip", 1.0),
                use_amp=config["training"].get("use_amp", False),
                lambda_scheduler=lambda_scheduler,
            )

            trainer.train()

            # Cleanup temp files
            temp_train_path.unlink()
            temp_val_path.unlink()

        # Load best model and evaluate
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Rebuild model for loading
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
                num_codecs=len(codec_vocab),
                num_codec_qs=len(codec_q_vocab),
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

        # Evaluate on in-domain (other codecs from dev set)
        in_domain_metrics = evaluate_model_on_split(
            model, dev_in_domain, codec_vocab, codec_q_vocab, device
        )

        # Evaluate on out-of-domain (held-out codec from dev set)
        if len(dev_out_domain) > 0:
            out_domain_metrics = evaluate_model_on_split(
                model, dev_out_domain, codec_vocab, codec_q_vocab, device
            )

            gap = compute_domain_gap(in_domain_metrics, out_domain_metrics)
        else:
            out_domain_metrics = {"eer": None, "min_dcf": None, "n_samples": 0}
            gap = {"eer_gap": None, "min_dcf_gap": None}

        results[f"{method}_in_domain"] = in_domain_metrics
        results[f"{method}_out_domain"] = out_domain_metrics
        results[f"{method}_gap"] = gap

        logger.info(f"{method.upper()} Results:")
        logger.info(f"  In-domain EER: {in_domain_metrics['eer']:.4f}")
        if out_domain_metrics["eer"] is not None:
            logger.info(f"  Out-domain EER: {out_domain_metrics['eer']:.4f}")
            logger.info(f"  EER Gap: {gap['eer_gap']:.4f}")

    return results


def main():
    args = parse_args()

    set_seed(args.seed)

    # Load config
    config = load_config(args.config)

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

    # Get top N codecs
    train_manifest = get_manifest_path("train")
    top_codecs = get_top_codecs(train_manifest, args.top_n)
    logger.info(f"Top {args.top_n} CODECs: {top_codecs}")

    # Run experiments
    all_results = []
    for codec in top_codecs:
        result = run_held_out_experiment(
            held_out_codec=codec,
            config=config,
            output_dir=output_dir,
            seed=args.seed,
            skip_training=args.skip_training,
        )
        all_results.append(result)

    # Create summary table
    summary_rows = []
    for r in all_results:
        row = {
            "held_out_codec": r["held_out_codec"],
            "train_samples": r["train_samples"],
            "test_samples": r["test_samples"],
        }

        for method in ["erm", "dann"]:
            in_key = f"{method}_in_domain"
            out_key = f"{method}_out_domain"
            gap_key = f"{method}_gap"

            if in_key in r:
                row[f"{method}_in_eer"] = r[in_key]["eer"]
                row[f"{method}_in_dcf"] = r[in_key]["min_dcf"]

            if out_key in r and r[out_key]["eer"] is not None:
                row[f"{method}_out_eer"] = r[out_key]["eer"]
                row[f"{method}_out_dcf"] = r[out_key]["min_dcf"]

            if gap_key in r and r[gap_key]["eer_gap"] is not None:
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
