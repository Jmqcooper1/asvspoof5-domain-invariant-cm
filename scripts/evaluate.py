#!/usr/bin/env python3
"""Evaluation entrypoint for trained models.

Usage:
    # Evaluate on dev set
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split dev

    # Evaluate with per-domain breakdown
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split dev --per-domain

    # Evaluate on eval set
    python scripts/evaluate.py --checkpoint runs/my_run/checkpoints/best.pt --split eval
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.data import ASVspoof5Dataset, AudioCollator, load_vocab
from asvspoof5_domain_invariant_cm.evaluation import (
    generate_overall_metrics,
    save_domain_tables,
    save_metrics_report,
    save_predictions,
    generate_scorefile,
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
from asvspoof5_domain_invariant_cm.utils import get_device, get_manifest_path, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "eval"],
        default="dev",
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: checkpoint parent dir)",
    )
    parser.add_argument(
        "--per-domain",
        action="store_true",
        help="Compute per-domain (CODEC, CODEC_Q) metrics",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--scorefile",
        action="store_true",
        help="Generate official-format score file",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Get vocab sizes from checkpoint or load from run dir
    run_dir = checkpoint_path.parent.parent
    codec_vocab = load_vocab(run_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(run_dir / "codec_q_vocab.json")

    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)

    # Build model
    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,  # Always freeze for inference
        layer_selection=layer_selection.get("method", "weighted"),
        k=layer_selection.get("k", 6),
        layer_indices=layer_selection.get("layers"),
        init_lower_bias=layer_selection.get("init_lower_bias", True),
    )

    pooling_method = pooling_cfg.get("method", "stats")
    pooling = create_pooling(pooling_method, backbone.hidden_size)

    if pooling_method == "stats":
        proj_input_dim = backbone.hidden_size * 2
    else:
        proj_input_dim = backbone.hidden_size

    projection = ProjectionHead(
        input_dim=proj_input_dim,
        hidden_dim=projection_cfg.get("hidden_dim", 512),
        output_dim=projection_cfg.get("output_dim", 256),
        num_layers=projection_cfg.get("num_layers", 2),
        dropout=projection_cfg.get("dropout", 0.1),
    )

    repr_dim = projection_cfg.get("output_dim", 256)

    task_head = ClassifierHead(
        input_dim=repr_dim,
        num_classes=classifier_cfg.get("num_classes", 2),
        hidden_dim=classifier_cfg.get("hidden_dim", 0),
        dropout=classifier_cfg.get("dropout", 0.1),
    )

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
            lambda_=0.0,  # No GRL effect during inference
        )
    else:
        model = ERMModel(
            backbone=backbone,
            pooling=pooling,
            projection=projection,
            task_head=task_head,
        )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, codec_vocab, codec_q_vocab


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[dict]:
    """Run inference and collect predictions.

    Args:
        model: Trained model.
        dataloader: Evaluation dataloader.
        device: Device.

    Returns:
        List of prediction dictionaries.
    """
    predictions = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        # Score convention: higher = more bonafide (class 0)
        probs = torch.softmax(outputs["task_logits"], dim=-1)
        scores = probs[:, 0]  # P(bonafide)
        preds = outputs["task_logits"].argmax(dim=-1)

        # Collect predictions
        metadata = batch["metadata"]
        batch_size = waveform.shape[0]

        for i in range(batch_size):
            pred_dict = {
                "flac_file": metadata["flac_file"][i],
                "score": scores[i].cpu().item(),
                "prediction": preds[i].cpu().item(),
                "y_task": batch["y_task"][i].item(),
                "y_codec": batch["y_codec"][i].item(),
                "y_codec_q": batch["y_codec_q"][i].item(),
            }

            # Add optional metadata
            for key in ["speaker_id", "codec_seed", "codec", "codec_q", "attack_label", "attack_tag"]:
                if key in metadata:
                    pred_dict[key] = metadata[key][i]

            predictions.append(pred_dict)

    return predictions


def main():
    args = parse_args()

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(
        args.checkpoint, device
    )

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.checkpoint.parent.parent / f"eval_{args.split}"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    audio_cfg = config.get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 16000)
    max_duration = audio_cfg.get("max_duration_sec", 6.0)

    dataset = ASVspoof5Dataset(
        manifest_path=get_manifest_path(args.split),
        codec_vocab=codec_vocab,
        codec_q_vocab=codec_q_vocab,
        max_duration_sec=max_duration,
        sample_rate=sample_rate,
        mode="eval",
    )

    logger.info(f"Evaluation samples: {len(dataset)}")

    # Create dataloader
    fixed_length = int(max_duration * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Run inference
    predictions = run_inference(model, dataloader, device)

    # Convert to DataFrame
    df = pd.DataFrame(predictions)

    # Save predictions
    pred_path = output_dir / "predictions.tsv"
    save_predictions(predictions, pred_path)
    logger.info(f"Saved predictions: {pred_path}")

    # Generate score file
    if args.scorefile:
        score_path = output_dir / f"scores_{args.split}.txt"
        generate_scorefile(df, score_path)
        logger.info(f"Saved score file: {score_path}")

    # Compute overall metrics
    scores = df["score"].values
    labels = df["y_task"].values

    metrics = generate_overall_metrics(
        scores, labels,
        bootstrap=args.bootstrap,
        seed=config.get("seed", 42),
    )

    logger.info("=" * 60)
    logger.info(f"Overall metrics ({args.split}):")
    logger.info(f"  EER: {metrics['eer']:.4f}")
    logger.info(f"  minDCF: {metrics['min_dcf']:.4f}")
    logger.info(f"  Samples: {metrics['n_samples']} (bonafide: {metrics['n_bonafide']}, spoof: {metrics['n_spoof']})")
    logger.info("=" * 60)

    # Save overall metrics
    save_metrics_report(metrics, output_dir / "metrics.json")

    # Per-domain breakdown
    if args.per_domain:
        tables_dir = output_dir / "tables"
        table_paths = save_domain_tables(df, tables_dir)

        for domain, path in table_paths.items():
            logger.info(f"Saved {domain} table: {path}")

            # Log summary
            domain_df = pd.read_csv(path)
            logger.info(f"\n{domain.upper()} breakdown:")
            logger.info(domain_df.to_string(index=False))

    logger.info(f"\nResults saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
