#!/usr/bin/env python3
"""Run layer-wise domain probes to analyze domain leakage.

This script trains linear classifiers on frozen representations from each
layer to predict domain labels (CODEC, CODEC_Q). Lower probe accuracy
indicates more domain-invariant representations.

Usage:
    # Single model
    python scripts/probe_domain.py --checkpoint runs/erm_run/checkpoints/best.pt

    # Compare ERM vs DANN
    python scripts/probe_domain.py \
        --erm-checkpoint runs/erm_run/checkpoints/best.pt \
        --dann-checkpoint runs/dann_run/checkpoints/best.pt

    # With wandb logging
    python scripts/probe_domain.py --erm-checkpoint ... --dann-checkpoint ... --wandb
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from asvspoof5_domain_invariant_cm.analysis import (
    compare_probe_accuracies,
    layerwise_probing,
)
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
    parser = argparse.ArgumentParser(description="Run domain probes")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to single model checkpoint",
    )
    parser.add_argument(
        "--erm-checkpoint",
        type=Path,
        default=None,
        help="Path to ERM model checkpoint (for comparison)",
    )
    parser.add_argument(
        "--dann-checkpoint",
        type=Path,
        default=None,
        help="Path to DANN model checkpoint (for comparison)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev", "eval"],
        default="dev",
        help="Data split to probe on",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to probe: 'all' or comma-separated indices",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="codec,codec_q",
        help="Domains to probe for (comma-separated)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logistic", "svm"],
        default="logistic",
        help="Probe classifier type",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
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
        default=5000,
        help="Number of samples to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
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
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layer_indices: list[int] = None,
    max_samples: int = None,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings from each layer.

    Args:
        model: Model to extract from.
        dataloader: Dataloader.
        device: Device.
        layer_indices: Specific layers to extract (None = all).
        max_samples: Maximum samples to use.

    Returns:
        Tuple of (layer_embeddings, codec_labels, codec_q_labels, repr_embeddings).
    """
    layer_embeddings = {}
    repr_embeddings = []
    all_codec = []
    all_codec_q = []

    n_samples = 0

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if max_samples and n_samples >= max_samples:
            break

        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)

        outputs = model(waveform, attention_mask, lengths)

        batch_size = waveform.shape[0]

        # Get all hidden states
        all_hidden_states = outputs["all_hidden_states"]

        for layer_idx, hidden_state in enumerate(all_hidden_states):
            if layer_indices is not None and layer_idx not in layer_indices:
                continue

            # Mean pool over time for each layer
            # hidden_state: [B, T, D]
            pooled = hidden_state.mean(dim=1).cpu().numpy()

            if layer_idx not in layer_embeddings:
                layer_embeddings[layer_idx] = []
            layer_embeddings[layer_idx].append(pooled)

        # Get projection repr
        repr_embeddings.append(outputs["repr"].cpu().numpy())

        # Get labels
        all_codec.append(batch["y_codec"].numpy())
        all_codec_q.append(batch["y_codec_q"].numpy())

        n_samples += batch_size

    # Concatenate
    for layer_idx in layer_embeddings:
        layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)
        if max_samples:
            layer_embeddings[layer_idx] = layer_embeddings[layer_idx][:max_samples]

    repr_embeddings = np.concatenate(repr_embeddings, axis=0)
    if max_samples:
        repr_embeddings = repr_embeddings[:max_samples]

    all_codec = np.concatenate(all_codec)[:max_samples] if max_samples else np.concatenate(all_codec)
    all_codec_q = np.concatenate(all_codec_q)[:max_samples] if max_samples else np.concatenate(all_codec_q)

    return layer_embeddings, all_codec, all_codec_q, repr_embeddings


def plot_probe_results(
    results: dict,
    output_path: Path,
    title: str = "Domain Probe Accuracy by Layer",
):
    """Plot probe accuracy vs layer."""
    plt.figure(figsize=(10, 6))

    layers = sorted([k for k in results["per_layer"].keys() if k != "repr"])
    accuracies = [results["per_layer"][l]["accuracy"] for l in layers]

    plt.plot(layers, accuracies, "o-", linewidth=2, markersize=8, label="Layers")

    # Add repr point if available
    if "repr" in results["per_layer"]:
        repr_acc = results["per_layer"]["repr"]["accuracy"]
        plt.axhline(y=repr_acc, color='r', linestyle='--', label=f'Repr ({repr_acc:.3f})')

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Probe Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison(
    erm_results: dict,
    dann_results: dict,
    output_path: Path,
    domain: str,
):
    """Plot ERM vs DANN probe comparison."""
    plt.figure(figsize=(10, 6))

    layers = sorted([k for k in erm_results["per_layer"].keys() if k != "repr"])
    erm_acc = [erm_results["per_layer"][l]["accuracy"] for l in layers]
    dann_acc = [dann_results["per_layer"][l]["accuracy"] for l in layers]

    plt.plot(layers, erm_acc, "o-", linewidth=2, markersize=8, label="ERM")
    plt.plot(layers, dann_acc, "s-", linewidth=2, markersize=8, label="DANN")

    # Add repr points if available
    if "repr" in erm_results["per_layer"]:
        erm_repr = erm_results["per_layer"]["repr"]["accuracy"]
        dann_repr = dann_results["per_layer"]["repr"]["accuracy"]
        max_layer = max(layers)
        plt.scatter([max_layer + 1], [erm_repr], marker='o', s=100, color='C0', edgecolor='black', zorder=5)
        plt.scatter([max_layer + 1], [dann_repr], marker='s', s=100, color='C1', edgecolor='black', zorder=5)
        plt.annotate('ERM repr', (max_layer + 1.2, erm_repr), fontsize=9)
        plt.annotate('DANN repr', (max_layer + 1.2, dann_repr), fontsize=9)

    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Probe Accuracy", fontsize=12)
    plt.title(f"{domain.upper()} Probe Accuracy: ERM vs DANN", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def log_to_wandb(
    args,
    all_results: dict,
    output_dir: Path,
    mode: str,
    comparison_results: dict = None,
) -> None:
    """Log probe results to wandb."""
    if not WANDB_AVAILABLE:
        logger.warning("Wandb not available, skipping logging")
        return

    try:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"probes_{mode}",
            job_type="analysis",
        )

        # Log per-model, per-domain results
        for model_name, model_results in all_results.items():
            for domain, results in model_results.items():
                # Log per-layer accuracies
                for layer, layer_data in results["per_layer"].items():
                    wandb.log({
                        f"probe/{model_name}/{domain}/layer_{layer}": layer_data["accuracy"],
                    })

                # Log summary stats
                wandb.log({
                    f"probe/{model_name}/{domain}/max_leakage_layer": results.get("max_leakage_layer"),
                    f"probe/{model_name}/{domain}/max_leakage_accuracy": results.get("max_leakage_accuracy"),
                })

        # Log comparison results
        if comparison_results:
            for domain, comparison in comparison_results.items():
                wandb.log({
                    f"probe_comparison/{domain}/avg_erm_accuracy": comparison["avg_erm_accuracy"],
                    f"probe_comparison/{domain}/avg_dann_accuracy": comparison["avg_dann_accuracy"],
                    f"probe_comparison/{domain}/avg_reduction": comparison["avg_reduction"],
                })

                # Log comparison table
                rows = []
                for layer, data in comparison["per_layer"].items():
                    rows.append([
                        str(layer),
                        data["erm_accuracy"],
                        data["dann_accuracy"],
                        data["reduction"],
                    ])
                table = wandb.Table(
                    columns=["layer", "erm_accuracy", "dann_accuracy", "reduction"],
                    data=rows,
                )
                wandb.log({f"probe_comparison/{domain}/table": table})

        # Log plots
        for plot_file in output_dir.glob("*.png"):
            wandb.log({f"plots/{plot_file.stem}": wandb.Image(str(plot_file))})

        wandb.finish()
        logger.info("Logged probe results to Wandb")

    except Exception as e:
        logger.warning(f"Wandb logging failed: {e}")


def main():
    args = parse_args()

    np.random.seed(args.seed)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Parse layer indices
    if args.layers == "all":
        layer_indices = None
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    domains = args.domains.split(",")

    # Determine mode
    if args.erm_checkpoint and args.dann_checkpoint:
        mode = "comparison"
        checkpoints = {
            "erm": args.erm_checkpoint,
            "dann": args.dann_checkpoint,
        }
    elif args.checkpoint:
        mode = "single"
        checkpoints = {"model": args.checkpoint}
    else:
        raise ValueError("Provide --checkpoint or both --erm-checkpoint and --dann-checkpoint")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        if mode == "comparison":
            output_dir = args.erm_checkpoint.parent.parent.parent / "probes_comparison"
        else:
            output_dir = args.checkpoint.parent.parent / "probes"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with JSON output
    setup_logging(output_dir, json_output=True)
    logger.info(f"Output directory: {output_dir}")

    # Log experiment context
    context = get_experiment_context()
    logger.info(f"Git commit: {context['git'].get('commit', 'N/A')[:8] if context['git'].get('commit') else 'N/A'}")

    # Load first model to get config/vocabs
    first_ckpt = list(checkpoints.values())[0]
    _, config, codec_vocab, codec_q_vocab = load_model_from_checkpoint(first_ckpt, device)

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

    logger.info(f"Using {len(dataset)} samples for probing")

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

    # Extract embeddings for each model
    all_results = {}

    for model_name, ckpt_path in checkpoints.items():
        logger.info(f"\nProcessing {model_name}: {ckpt_path}")

        model, _, _, _ = load_model_from_checkpoint(ckpt_path, device)

        layer_embeddings, codec_labels, codec_q_labels, repr_embeddings = extract_embeddings(
            model, dataloader, device, layer_indices, args.n_samples
        )

        # Add projection repr as a special "layer"
        layer_embeddings["repr"] = repr_embeddings

        model_results = {}

        for domain in domains:
            logger.info(f"  Probing {domain}...")

            if domain == "codec":
                labels = codec_labels
            else:
                labels = codec_q_labels

            results = layerwise_probing(
                layer_embeddings,
                labels,
                classifier=args.classifier,
                cv_folds=args.cv_folds,
                seed=args.seed,
            )

            model_results[domain] = results

            # Plot
            plot_path = output_dir / f"{model_name}_{domain}_probes.png"
            plot_probe_results(
                results, plot_path,
                title=f"{model_name.upper()} - {domain.upper()} Probe Accuracy"
            )

            logger.info(f"    Max leakage layer: {results['max_leakage_layer']}")
            logger.info(f"    Max accuracy: {results['max_leakage_accuracy']:.4f}")
            if "repr" in results["per_layer"]:
                logger.info(f"    Repr accuracy: {results['per_layer']['repr']['accuracy']:.4f}")

        all_results[model_name] = model_results

    # Save results
    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    logger.info(f"Saved results: {results_path}")

    # If comparison mode, create comparison plots and CSV
    comparison_results = {}
    if mode == "comparison":
        for domain in domains:
            erm_results = all_results["erm"][domain]
            dann_results = all_results["dann"][domain]

            # Comparison plot
            plot_path = output_dir / f"comparison_{domain}_probes.png"
            plot_comparison(erm_results, dann_results, plot_path, domain)

            # Comparison CSV
            comparison = compare_probe_accuracies(erm_results, dann_results)
            comparison_results[domain] = comparison

            rows = []
            for layer, data in comparison["per_layer"].items():
                rows.append({
                    "layer": layer,
                    "erm_accuracy": data["erm_accuracy"],
                    "dann_accuracy": data["dann_accuracy"],
                    "reduction": data["reduction"],
                    "relative_reduction": data["relative_reduction"],
                })

            df = pd.DataFrame(rows)
            csv_path = output_dir / f"comparison_{domain}.csv"
            df.to_csv(csv_path, index=False)

            logger.info(f"\n{domain.upper()} comparison:")
            logger.info(f"  Avg ERM accuracy: {comparison['avg_erm_accuracy']:.4f}")
            logger.info(f"  Avg DANN accuracy: {comparison['avg_dann_accuracy']:.4f}")
            logger.info(f"  Avg reduction: {comparison['avg_reduction']:.4f}")

    # Build analysis complete wide event
    analysis_event = {
        "analysis_type": "domain_probes",
        "mode": mode,
        "split": args.split,
        "n_samples": len(dataset),
        "domains": domains,
        "classifier": args.classifier,
        "cv_folds": args.cv_folds,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if comparison_results:
        analysis_event["comparison_summary"] = {
            domain: {
                "avg_erm_accuracy": comp["avg_erm_accuracy"],
                "avg_dann_accuracy": comp["avg_dann_accuracy"],
                "avg_reduction": comp["avg_reduction"],
            }
            for domain, comp in comparison_results.items()
        }

    # Save analysis event
    with open(output_dir / "analysis_event.json", "w") as f:
        json.dump(analysis_event, f, indent=2, default=str)

    logger.info(f"\nAll results saved to: {output_dir}")

    # Wandb logging
    if args.wandb:
        log_to_wandb(args, all_results, output_dir, mode, comparison_results)

    return 0


if __name__ == "__main__":
    exit(main())
