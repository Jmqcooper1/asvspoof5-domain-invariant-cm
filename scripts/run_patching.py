#!/usr/bin/env python3
"""Run activation patching experiments.

This script performs limited activation patching to test causal effects
of domain-heavy components on detection and domain leakage.

Usage:
    python scripts/run_patching.py \
        --source runs/dann_run/checkpoints/best.pt \
        --target runs/erm_run/checkpoints/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.data import ASVspoof5Dataset, AudioCollator, load_vocab
from asvspoof5_domain_invariant_cm.evaluation import compute_eer
from asvspoof5_domain_invariant_cm.models import (
    ClassifierHead,
    DANNModel,
    ERMModel,
    MultiHeadDomainDiscriminator,
    ProjectionHead,
    create_backbone,
    create_pooling,
)
from asvspoof5_domain_invariant_cm.utils import get_device, get_manifest_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation patching")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source model checkpoint (DANN - domain invariant)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target model checkpoint (ERM - baseline)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split",
    )
    parser.add_argument(
        "--patch-type",
        type=str,
        choices=["repr", "layer"],
        default="repr",
        help="What to patch: 'repr' (projection output) or 'layer' (transformer layers)",
    )
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="10,11",
        help="Layer indices to patch (comma-separated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for patching",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    run_dir = checkpoint_path.parent.parent
    codec_vocab = load_vocab(run_dir / "codec_vocab.json")
    codec_q_vocab = load_vocab(run_dir / "codec_q_vocab.json")

    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)

    backbone_cfg = config.get("backbone", {})
    projection_cfg = config.get("projection", {})
    classifier_cfg = config.get("classifier", {})
    pooling_cfg = config.get("pooling", {})
    training_cfg = config.get("training", {})

    layer_selection = backbone_cfg.get("layer_selection", {})
    backbone = create_backbone(
        name=backbone_cfg.get("name", "wavlm_base_plus"),
        pretrained=backbone_cfg.get("pretrained", "microsoft/wavlm-base-plus"),
        freeze=True,
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
            lambda_=0.0,
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
    model.eval()

    return model, config, codec_vocab, codec_q_vocab


@torch.no_grad()
def run_repr_patching(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = None,
) -> dict:
    """Patch projection representations from source to target.

    Args:
        source_model: Source model (DANN).
        target_model: Target model (ERM).
        dataloader: DataLoader.
        device: Device.
        max_samples: Maximum samples.

    Returns:
        Patching results.
    """
    baseline_scores = []
    patched_scores = []
    all_labels = []
    n_samples = 0

    for batch in tqdm(dataloader, desc="Patching repr"):
        if max_samples and n_samples >= max_samples:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        y_task = batch["y_task"]

        batch_size = waveform.shape[0]

        # Get source (DANN) representation
        source_outputs = source_model(waveform, attention_mask, lengths)
        source_repr = source_outputs["repr"]

        # Get baseline target (ERM) outputs
        target_outputs = target_model(waveform, attention_mask, lengths)
        baseline_logits = target_outputs["task_logits"]

        # Patch: replace target repr with source repr and re-run task head
        patched_logits = target_model.task_head(source_repr)

        # Scores: P(bonafide)
        baseline_probs = torch.softmax(baseline_logits, dim=-1)[:, 0]
        patched_probs = torch.softmax(patched_logits, dim=-1)[:, 0]

        baseline_scores.append(baseline_probs.cpu().numpy())
        patched_scores.append(patched_probs.cpu().numpy())
        all_labels.append(y_task.numpy())

        n_samples += batch_size

    baseline_scores = np.concatenate(baseline_scores)[:max_samples]
    patched_scores = np.concatenate(patched_scores)[:max_samples]
    all_labels = np.concatenate(all_labels)[:max_samples]

    # Compute metrics
    baseline_eer, _ = compute_eer(baseline_scores, all_labels)
    patched_eer, _ = compute_eer(patched_scores, all_labels)

    # Score change
    score_change = patched_scores - baseline_scores

    return {
        "baseline_eer": float(baseline_eer),
        "patched_eer": float(patched_eer),
        "eer_change": float(patched_eer - baseline_eer),
        "mean_score_change": float(np.mean(score_change)),
        "std_score_change": float(np.std(score_change)),
        "n_samples": len(all_labels),
    }


@torch.no_grad()
def run_layer_patching(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layer_indices: list[int],
    max_samples: int = None,
) -> dict:
    """Patch specific layer outputs from source to target.

    This is more complex because we need to intervene in the forward pass.
    """
    results = {}

    for layer_idx in layer_indices:
        logger.info(f"  Patching layer {layer_idx}...")

        baseline_scores = []
        patched_scores = []
        all_labels = []
        n_samples = 0

        for batch in tqdm(dataloader, desc=f"Layer {layer_idx}", leave=False):
            if max_samples and n_samples >= max_samples:
                break

            waveform = batch["waveform"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lengths = batch["lengths"].to(device)
            y_task = batch["y_task"]

            batch_size = waveform.shape[0]

            # Get source hidden states
            source_outputs = source_model(waveform, attention_mask, lengths)
            source_hidden_states = source_outputs["all_hidden_states"]

            # Get target hidden states and baseline
            target_outputs = target_model(waveform, attention_mask, lengths)
            baseline_logits = target_outputs["task_logits"]

            # For patching, we need to manually reconstruct the forward pass
            # with the patched hidden state
            if layer_idx < len(source_hidden_states):
                # Get the source layer output
                source_layer_output = source_hidden_states[layer_idx]

                # Replace in target's pipeline
                # This is a simplified version - full implementation would
                # require hooks or manual forward pass reconstruction

                # For now, we approximate by replacing the mixed output
                # if we're patching the last layer being used
                target_hidden_states = target_outputs["all_hidden_states"]

                # Simple approach: patch and re-run pooling + projection + head
                # We'll use the source layer output for the selected layer
                # and recompute downstream

                # Get layer weights from target backbone
                selected_layers = list(range(len(target_hidden_states)))
                if hasattr(target_model.backbone, 'k'):
                    k = target_model.backbone.k
                    if target_model.backbone.layer_selection == "last_k":
                        selected_layers = list(range(len(target_hidden_states) - k, len(target_hidden_states)))
                    elif target_model.backbone.layer_selection == "first_k":
                        selected_layers = list(range(k))

                # Create patched hidden states
                patched_hidden_states = list(target_hidden_states)
                patched_hidden_states[layer_idx] = source_layer_output

                # Recompute layer mix
                weights = torch.softmax(target_model.backbone.layer_pooling.weights, dim=0)

                # Select layers
                if target_model.backbone.layer_selection == "first_k":
                    selected = patched_hidden_states[:target_model.backbone.k]
                elif target_model.backbone.layer_selection == "last_k":
                    selected = patched_hidden_states[-target_model.backbone.k:]
                else:
                    selected = patched_hidden_states

                stacked = torch.stack(selected, dim=0)
                patched_mixed = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)

                # Pooling
                patched_pooled = target_model.pooling(patched_mixed, lengths)

                # Projection
                patched_repr = target_model.projection(patched_pooled)

                # Task head
                patched_logits = target_model.task_head(patched_repr)
            else:
                patched_logits = baseline_logits

            # Scores
            baseline_probs = torch.softmax(baseline_logits, dim=-1)[:, 0]
            patched_probs = torch.softmax(patched_logits, dim=-1)[:, 0]

            baseline_scores.append(baseline_probs.cpu().numpy())
            patched_scores.append(patched_probs.cpu().numpy())
            all_labels.append(y_task.numpy())

            n_samples += batch_size

        baseline_scores = np.concatenate(baseline_scores)[:max_samples]
        patched_scores = np.concatenate(patched_scores)[:max_samples]
        all_labels = np.concatenate(all_labels)[:max_samples]

        baseline_eer, _ = compute_eer(baseline_scores, all_labels)
        patched_eer, _ = compute_eer(patched_scores, all_labels)

        score_change = patched_scores - baseline_scores

        results[f"layer_{layer_idx}"] = {
            "baseline_eer": float(baseline_eer),
            "patched_eer": float(patched_eer),
            "eer_change": float(patched_eer - baseline_eer),
            "mean_score_change": float(np.mean(score_change)),
            "std_score_change": float(np.std(score_change)),
        }

    return results


def main():
    args = parse_args()

    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.target.parent.parent.parent / "patching_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load models
    logger.info(f"Loading source (DANN): {args.source}")
    source_model, _, codec_vocab, codec_q_vocab = load_model_from_checkpoint(
        args.source, device
    )

    logger.info(f"Loading target (ERM): {args.target}")
    target_model, config, _, _ = load_model_from_checkpoint(args.target, device)

    # Create dataset
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

    # Limit samples
    if args.n_samples and args.n_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)

    logger.info(f"Using {len(dataset)} samples for patching")

    fixed_length = int(max_duration * sample_rate)
    collator = AudioCollator(fixed_length=fixed_length, mode="eval")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
    )

    results = {
        "source": str(args.source),
        "target": str(args.target),
        "split": args.split,
        "n_samples": len(dataset),
        "patch_type": args.patch_type,
    }

    # Run patching
    if args.patch_type == "repr":
        logger.info("Running representation patching (DANN repr -> ERM)...")
        repr_results = run_repr_patching(
            source_model, target_model, dataloader, device, args.n_samples
        )
        results["repr_patching"] = repr_results

        logger.info("\nRepresentation patching results:")
        logger.info(f"  Baseline EER: {repr_results['baseline_eer']:.4f}")
        logger.info(f"  Patched EER: {repr_results['patched_eer']:.4f}")
        logger.info(f"  EER change: {repr_results['eer_change']:.4f}")
        logger.info(f"  Mean score change: {repr_results['mean_score_change']:.4f}")

    elif args.patch_type == "layer":
        layer_indices = [int(x) for x in args.layer_indices.split(",")]
        logger.info(f"Running layer patching for layers {layer_indices}...")

        layer_results = run_layer_patching(
            source_model, target_model, dataloader, device, layer_indices, args.n_samples
        )
        results["layer_patching"] = layer_results

        logger.info("\nLayer patching results:")
        for layer, layer_data in layer_results.items():
            logger.info(f"  {layer}:")
            logger.info(f"    Baseline EER: {layer_data['baseline_eer']:.4f}")
            logger.info(f"    Patched EER: {layer_data['patched_eer']:.4f}")
            logger.info(f"    EER change: {layer_data['eer_change']:.4f}")

    # Save results
    output_path = output_dir / "patching_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
