#!/usr/bin/env python3
"""Compute CKA (Centered Kernel Alignment) between ERM and DANN representations.

This script computes CKA similarity between ERM and DANN model representations
to analyze how domain-adversarial training changes the learned features.

Usage:
    python scripts/run_cka.py \
        --erm-checkpoint runs/erm_run/checkpoints/best.pt \
        --dann-checkpoint runs/dann_run/checkpoints/best.pt

    # With wandb logging
    python scripts/run_cka.py \
        --erm-checkpoint ... --dann-checkpoint ... --wandb
"""

import argparse
import json
import logging
from datetime import datetime
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
from asvspoof5_domain_invariant_cm.utils import (
    get_device,
    get_experiment_context,
    get_manifest_path,
    setup_logging,
)

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
        default=None,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (default: from config)",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["hidden_states", "mixed", "repr", "layer_contrib"],
        default="layer_contrib",
        help="Representation to compare for CKA",
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Wandb",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="asvspoof5-dann",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (team or username)",
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

        # Discriminator taps pre-projection pooled features (e.g. 1536-dim stats pooling)
        disc_input_dim = disc_cfg.get("input_dim", proj_input_dim)
        domain_discriminator = MultiHeadDomainDiscriminator(
            input_dim=disc_input_dim,
            num_codecs=num_codecs,
            num_codec_qs=num_codec_qs,
            hidden_dim=disc_cfg.get("hidden_dim", 512),
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
def _format_layer_label(layer) -> str:
    if isinstance(layer, int):
        return f"L{layer}"
    return str(layer)


def _resolve_selected_layer_indices(
    total_layers: int,
    method: str,
    k: int,
    layer_indices: list[int] | None,
) -> list[int]:
    if method == "first_k":
        return list(range(min(k, total_layers)))
    if method == "last_k":
        start = max(total_layers - k, 0)
        return list(range(start, total_layers))
    if method == "specific" and layer_indices:
        return [i for i in layer_indices if i < total_layers]
    return list(range(total_layers))


def _get_layer_weights(
    model: torch.nn.Module,
    selected_indices: list[int],
) -> torch.Tensor:
    weights = torch.softmax(model.backbone.layer_pooling.weights, dim=0)
    if weights.numel() != len(selected_indices):
        logger.warning(
            "Layer pooling weights size does not match selected layers; "
            "falling back to uniform weights."
        )
        weights = torch.ones(len(selected_indices), device=weights.device) / len(
            selected_indices
        )
    return weights


@torch.no_grad()
def extract_representation_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = None,
    representation: str = "layer_contrib",
) -> dict:
    """Extract embeddings for a specified representation."""
    embeddings = {}
    n_samples = 0
    selected_indices = None
    weights = None

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if max_samples and n_samples >= max_samples:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)
        batch_size = waveform.shape[0]

        all_hidden_states = outputs["all_hidden_states"]

        if representation == "hidden_states":
            for layer_idx, hidden_state in enumerate(all_hidden_states):
                pooled = hidden_state.mean(dim=1).cpu().numpy()
                embeddings.setdefault(layer_idx, []).append(pooled)
        elif representation == "repr":
            embeddings.setdefault("repr", []).append(outputs["repr"].cpu().numpy())
        else:
            if selected_indices is None:
                total_layers = len(all_hidden_states)
                selection_cfg = model.backbone
                selected_indices = _resolve_selected_layer_indices(
                    total_layers=total_layers,
                    method=selection_cfg.layer_selection,
                    k=selection_cfg.k,
                    layer_indices=selection_cfg.layer_indices,
                )
                weights = _get_layer_weights(model, selected_indices)

            selected_states = [all_hidden_states[i] for i in selected_indices]
            if representation == "mixed":
                stacked = torch.stack(selected_states, dim=0)
                mixed = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
                pooled = mixed.mean(dim=1).cpu().numpy()
                embeddings.setdefault("mixed", []).append(pooled)
            elif representation == "layer_contrib":
                for idx, layer_idx in enumerate(selected_indices):
                    pooled = selected_states[idx].mean(dim=1) * weights[idx]
                    embeddings.setdefault(layer_idx, []).append(pooled.cpu().numpy())
            else:
                raise ValueError(f"Unknown representation: {representation}")

        n_samples += batch_size

    for key in embeddings:
        embeddings[key] = np.concatenate(embeddings[key], axis=0)
        if max_samples:
            embeddings[key] = embeddings[key][:max_samples]

    return embeddings


def plot_cka_heatmap(
    cka_results: dict,
    output_path: Path,
):
    """Plot CKA similarity heatmap."""
    layers = sorted(cka_results["per_layer"].keys())
    cka_values = [cka_results["per_layer"][l]["cka"] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 3))

    # Create heatmap as a single row
    data = np.array(cka_values).reshape(1, -1)

    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([_format_layer_label(l) for l in layers])
    ax.set_yticks([])
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title("CKA Similarity (ERM vs DANN)", fontsize=14)

    # Add values
    for i, v in enumerate(cka_values):
        color = "white" if v < 0.5 else "black"
        ax.text(i, 0, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="CKA")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cka_line(
    cka_results: dict,
    output_path: Path,
):
    """Plot CKA similarity as line plot."""
    layers = sorted(cka_results["per_layer"].keys())
    cka_values = [cka_results["per_layer"][l]["cka"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, cka_values, "o-", linewidth=2, markersize=8, color="steelblue")
    ax.fill_between(layers, cka_values, alpha=0.3, color="steelblue")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("CKA Similarity", fontsize=12)
    ax.set_title("CKA Similarity (ERM vs DANN) by Layer", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Annotate min
    min_idx = np.argmin(cka_values)
    ax.annotate(
        f"Min: {cka_values[min_idx]:.3f}",
        (layers[min_idx], cka_values[min_idx]),
        textcoords="offset points",
        xytext=(0, -20),
        ha="center",
        fontsize=10,
        color="red",
    )

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
        xticklabels=[_format_layer_label(l) for l in layers],
        yticklabels=[_format_layer_label(l) for l in layers],
        ax=ax,
    )

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def log_to_wandb(
    args,
    cka_results: dict,
    output_dir: Path,
) -> None:
    """Log CKA results to wandb."""
    if not WANDB_AVAILABLE:
        logger.warning("Wandb not available, skipping logging")
        return

    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="cka_analysis",
            job_type="analysis",
        )

        # Log per-layer CKA
        for layer, data in cka_results["per_layer"].items():
            wandb.log({f"cka/layer_{layer}": data["cka"]})

        # Log summary stats
        wandb.log({
            "cka/mean": cka_results["mean_cka"],
            "cka/min": cka_results["min_cka"],
            "cka/max": cka_results["max_cka"],
            "cka/most_different_layer": cka_results["most_different_layer"],
        })

        # Log per-layer table
        rows = []
        for layer, data in cka_results["per_layer"].items():
            rows.append([layer, data["cka"]])
        table = wandb.Table(columns=["layer", "cka"], data=rows)
        wandb.log({"cka/per_layer_table": table})

        # Log plots
        for plot_file in output_dir.glob("*.png"):
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})

        wandb.finish()
        logger.info("Logged CKA results to Wandb")

    except Exception as e:
        logger.warning(f"Wandb logging failed: {e}")


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

    # Setup logging with JSON output
    setup_logging(output_dir, json_output=True)
    logger.info(f"Output directory: {output_dir}")

    # Log experiment context
    context = get_experiment_context()
    logger.info(f"Git commit: {context['git'].get('commit', 'N/A')[:8] if context['git'].get('commit') else 'N/A'}")

    # Load ERM model
    logger.info(f"Loading ERM model: {args.erm_checkpoint}")
    erm_model, config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(
        args.erm_checkpoint, device
    )

    # Load DANN model
    logger.info(f"Loading DANN model: {args.dann_checkpoint}")
    dann_model, _, _, _ = load_model_from_checkpoint(args.dann_checkpoint, device)

    if args.representation == "hidden_states":
        logger.warning(
            "Using hidden_states with a frozen SSL backbone will yield CKA~1.0. "
            "Consider --representation layer_contrib, mixed, or repr for a meaningful comparison."
        )

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

    dataloader_cfg = config.get("dataloader", {})
    batch_size = (
        int(args.batch_size)
        if args.batch_size is not None
        else int(dataloader_cfg.get("batch_size", 32))
    )
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(dataloader_cfg.get("num_workers", 4))
    )
    pin_memory = bool(dataloader_cfg.get("pin_memory", True))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
    )

    # Extract embeddings from both models
    logger.info("Extracting ERM embeddings...")
    erm_embeddings = extract_representation_embeddings(
        erm_model,
        dataloader,
        device,
        args.n_samples,
        representation=args.representation,
    )

    logger.info("Extracting DANN embeddings...")
    dann_embeddings = extract_representation_embeddings(
        dann_model,
        dataloader,
        device,
        args.n_samples,
        representation=args.representation,
    )

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

    # Plot CKA line plot
    plot_cka_line(cka_results, output_dir / "cka_erm_vs_dann_line.png")
    logger.info(f"Saved CKA line plot: {output_dir / 'cka_erm_vs_dann_line.png'}")

    # Save per-layer CKA to CSV
    rows = []
    for layer, data in cka_results["per_layer"].items():
        rows.append({
            "layer": str(layer),
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

    # Build analysis complete wide event
    analysis_event = {
        "analysis_type": "cka",
        "erm_checkpoint": str(args.erm_checkpoint),
        "dann_checkpoint": str(args.dann_checkpoint),
        "split": args.split,
        "n_samples": len(dataset),
        "mean_cka": cka_results["mean_cka"],
        "min_cka": cka_results["min_cka"],
        "max_cka": cka_results["max_cka"],
        "most_different_layer": cka_results["most_different_layer"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

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
        "analysis_event": analysis_event,
    }

    with open(output_dir / "cka_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAll results saved to: {output_dir}")

    # Wandb logging
    if args.wandb:
        log_to_wandb(args, cka_results, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
