#!/usr/bin/env python3
"""Compute CKA (Centered Kernel Alignment) between ERM and DANN representations.

Usage:
    python scripts/run_cka.py \
        --erm-checkpoint runs/erm_run/checkpoints/best.pt \
        --dann-checkpoint runs/dann_run/checkpoints/best.pt
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.analysis import compare_representations, layerwise_cka_matrix
from asvspoof5_domain_invariant_cm.data import ASVspoof5Dataset, AudioCollator, load_vocab
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
    parser = argparse.ArgumentParser(description="Compute CKA between ERM and DANN")
    parser.add_argument(
        "--erm-checkpoint",
        type=Path,
        required=True,
        help="Path to ERM model checkpoint",
    )
    parser.add_argument(
        "--dann-checkpoint",
        type=Path,
        required=True,
        help="Path to DANN model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split",
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
        default=2000,
        help="Number of samples for CKA computation",
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
def extract_layer_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = None,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled embeddings from each layer."""
    layer_embeddings = {}
    n_samples = 0

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if max_samples and n_samples >= max_samples:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)
        batch_size = waveform.shape[0]

        all_hidden_states = outputs["all_hidden_states"]

        for layer_idx, hidden_state in enumerate(all_hidden_states):
            pooled = hidden_state.mean(dim=1).cpu().numpy()

            if layer_idx not in layer_embeddings:
                layer_embeddings[layer_idx] = []
            layer_embeddings[layer_idx].append(pooled)

        n_samples += batch_size

    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)
        if max_samples:
            layer_embeddings[layer_idx] = layer_embeddings[layer_idx][:max_samples]

    return layer_embeddings


def plot_cka_heatmap(
    cka_results: dict,
    output_path: Path,
):
    """Plot CKA similarity heatmap."""
    layers = sorted(cka_results["per_layer"].keys())
    cka_values = [cka_results["per_layer"][l]["cka"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 3))

    # Create heatmap as a single row
    data = np.array(cka_values).reshape(1, -1)

    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticks([])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title("CKA Similarity (ERM vs DANN)", fontsize=14)

    # Add values
    for i, v in enumerate(cka_values):
        ax.text(i, 0, f"{v:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    layers: list[int],
    output_path: Path,
    title: str = "Layer-wise CKA Matrix",
):
    """Plot CKA matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cka_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=0,
        vmax=1,
        xticklabels=[f"L{l}" for l in layers],
        yticklabels=[f"L{l}" for l in layers],
        ax=ax,
    )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.erm_checkpoint.parent.parent.parent / "cka_analysis"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load ERM model
    logger.info(f"Loading ERM model: {args.erm_checkpoint}")
    erm_model, config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(
        args.erm_checkpoint, device
    )

    # Load DANN model
    logger.info(f"Loading DANN model: {args.dann_checkpoint}")
    dann_model, _, _, _ = load_model_from_checkpoint(args.dann_checkpoint, device)

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

    logger.info(f"Using {len(dataset)} samples for CKA")

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

    # Extract embeddings from both models
    logger.info("Extracting ERM embeddings...")
    erm_embeddings = extract_layer_embeddings(erm_model, dataloader, device, args.n_samples)

    logger.info("Extracting DANN embeddings...")
    dann_embeddings = extract_layer_embeddings(dann_model, dataloader, device, args.n_samples)

    # Compute CKA between ERM and DANN
    logger.info("Computing CKA...")
    cka_results = compare_representations(erm_embeddings, dann_embeddings)

    logger.info("\nCKA Results:")
    logger.info(f"  Mean CKA: {cka_results['mean_cka']:.4f}")
    logger.info(f"  Min CKA: {cka_results['min_cka']:.4f} (layer {cka_results['most_different_layer']})")
    logger.info(f"  Max CKA: {cka_results['max_cka']:.4f}")

    # Plot CKA heatmap
    plot_cka_heatmap(cka_results, output_dir / "cka_erm_vs_dann.png")
    logger.info(f"Saved CKA heatmap: {output_dir / 'cka_erm_vs_dann.png'}")

    # Save per-layer CKA to CSV
    rows = []
    for layer, data in cka_results["per_layer"].items():
        rows.append({
            "layer": layer,
            "cka": data["cka"],
            "erm_shape": str(data["erm_shape"]),
            "dann_shape": str(data["dann_shape"]),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "cka_per_layer.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CKA CSV: {csv_path}")

    # Compute within-model CKA matrices
    logger.info("Computing within-model CKA matrices...")

    erm_cka_matrix = layerwise_cka_matrix(erm_embeddings)
    dann_cka_matrix = layerwise_cka_matrix(dann_embeddings)

    layers = sorted(erm_embeddings.keys())

    plot_cka_matrix(erm_cka_matrix, layers, output_dir / "cka_matrix_erm.png", "ERM Layer-wise CKA")
    plot_cka_matrix(dann_cka_matrix, layers, output_dir / "cka_matrix_dann.png", "DANN Layer-wise CKA")

    # Save full results
    results = {
        "erm_vs_dann": {
            "mean_cka": cka_results["mean_cka"],
            "min_cka": cka_results["min_cka"],
            "max_cka": cka_results["max_cka"],
            "most_different_layer": cka_results["most_different_layer"],
            "per_layer": {
                str(k): {"cka": v["cka"]}
                for k, v in cka_results["per_layer"].items()
            },
        },
        "n_samples": len(dataset),
        "split": args.split,
    }

    with open(output_dir / "cka_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
