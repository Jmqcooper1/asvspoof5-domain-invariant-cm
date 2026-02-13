#!/usr/bin/env python3
"""Generate F2: OOD Gap visualization for thesis.

This script visualizes the in-domain (dev) vs out-of-domain (eval) 
generalization gap, showing how DANN reduces the gap compared to ERM.

Usage:
    python scripts/plot_ood_gap.py \\
        --input results/main_results.json \\
        --output figures/ood_gap.png \\
        --verbose

    # With demo data for testing
    python scripts/plot_ood_gap.py --demo --output figures/ood_gap.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_RUN_DIR = {
    "wavlm_erm": "wavlm_erm",
    "wavlm_dann": "wavlm_dann",
    "w2v2_erm": "w2v2_erm",
    "w2v2_dann": "w2v2_dann",
    "w2v2_dann_v2": "w2v2_dann_v2",
    "lfcc_gmm": "lfcc_gmm_32",
    "trillsson_logistic": "trillsson_logistic",
    "trillsson_mlp": "trillsson_mlp",
}

MODEL_LABELS = {
    "wavlm_erm": "WavLM ERM",
    "wavlm_dann": "WavLM DANN",
    "w2v2_erm": "Wav2Vec2 ERM",
    "w2v2_dann": "Wav2Vec2 DANN v1",
    "w2v2_dann_v2": "Wav2Vec2 DANN v2",
    "lfcc_gmm": "LFCC-GMM",
    "trillsson_logistic": "TRILLsson Logistic",
    "trillsson_mlp": "TRILLsson MLP",
}

MODEL_ORDER = list(MODEL_RUN_DIR.keys())


# ---------------------------------------------------------------------------
# Style Configuration (matching plot_rq3_combined.py)
# ---------------------------------------------------------------------------
COLORS = {
    "wavlm": "#4C72B0",      # Steel blue
    "w2v2": "#DD8452",       # Coral/orange
    "erm": "#E57373",        # Light red/coral
    "dann": "#64B5F6",       # Light blue
    "dev": "#90CAF9",        # Light blue (in-domain)
    "eval": "#EF9A9A",       # Light red (out-of-domain)
    "gap_arrow": "#333333",  # Dark gray
}

STYLE_CONFIG = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_main_results(path: Path) -> Dict[str, Any]:
    """Load main results data.
    
    Expected format:
    {
        "wavlm_erm": {"dev_eer": 0.05, "eval_eer": 0.08, ...},
        "wavlm_dann": {"dev_eer": 0.04, "eval_eer": 0.06, ...},
        "w2v2_erm": {...},
        "w2v2_dann": {...}
    }
    """
    logger.info(f"Loading main results from: {path}")
    
    with open(path) as f:
        data = json.load(f)
    
    return data


def get_first_numeric_value(data: Dict[str, Any], keys: list[str]) -> Optional[float]:
    for key in keys:
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    return None


def extract_dev_eer_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    direct_value = get_first_numeric_value(payload, ["val_eer", "dev_eer", "eer"])
    if direct_value is not None:
        return direct_value
    final_val = payload.get("final_val")
    if isinstance(final_val, dict) and final_val.get("eer") is not None:
        try:
            return float(final_val["eer"])
        except (TypeError, ValueError):
            pass
    return get_first_numeric_value(payload, ["best_eer"])


def load_main_results_from_runs(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load eval+dev metrics from results/runs model directories."""
    results: Dict[str, Dict[str, float]] = {}
    for model_key in MODEL_ORDER:
        run_dir = MODEL_RUN_DIR[model_key]
        eval_metrics_path = results_dir / run_dir / "eval_eval" / "metrics.json"
        if not eval_metrics_path.exists():
            continue

        eval_payload = load_main_results(eval_metrics_path)
        eval_eer = get_first_numeric_value(eval_payload, ["eer", "eval_eer"])
        eval_mindcf = get_first_numeric_value(eval_payload, ["min_dcf", "eval_mindcf"])
        model_entry: Dict[str, float] = {}
        if eval_eer is not None:
            model_entry["eval_eer"] = eval_eer
        if eval_mindcf is not None:
            model_entry["eval_mindcf"] = eval_mindcf

        dev_candidate_paths = [
            results_dir / run_dir / "eval_dev" / "metrics.json",
            results_dir / run_dir / "metrics.json",
            results_dir / run_dir / "metrics_train.json",
        ]
        for dev_path in dev_candidate_paths:
            if not dev_path.exists():
                continue
            dev_payload = load_main_results(dev_path)
            dev_eer = extract_dev_eer_from_payload(dev_payload)
            if dev_eer is not None:
                model_entry["dev_eer"] = dev_eer
                break

        results[model_key] = model_entry

    return results


def generate_demo_data() -> Dict[str, Any]:
    """Generate demo data for testing."""
    logger.info("Generating demo data")
    
    return {
        "wavlm_erm": {
            "dev_eer": 0.035,
            "eval_eer": 0.082,
            "eval_mindcf": 0.245,
        },
        "wavlm_dann": {
            "dev_eer": 0.038,
            "eval_eer": 0.065,
            "eval_mindcf": 0.198,
        },
        "w2v2_erm": {
            "dev_eer": 0.045,
            "eval_eer": 0.095,
            "eval_mindcf": 0.285,
        },
        "w2v2_dann": {
            "dev_eer": 0.048,
            "eval_eer": 0.078,
            "eval_mindcf": 0.235,
        },
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_ood_gap_bars(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create paired bar chart showing dev vs eval EER with gap annotations.
    
    Layout: Groups by backbone, then ERM vs DANN within each backbone.
    Each group has 2 bars (dev and eval) with connecting arrows showing the gap.
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    available_keys = [model_key for model_key in MODEL_ORDER if model_key in data]
    groups = [(MODEL_LABELS.get(model_key, model_key), model_key) for model_key in available_keys]
    if not groups:
        raise ValueError("No model data available for OOD gap bars")
    
    n_groups = len(groups)
    x = np.arange(n_groups)
    width = 0.35
    
    # Extract data
    dev_eers = []
    eval_eers = []
    gaps = []
    
    for _, key in groups:
        model_data = data.get(key, {})
        dev_raw = model_data.get("dev_eer")
        eval_raw = model_data.get("eval_eer")
        dev_eer = (float(dev_raw) * 100.0) if dev_raw is not None else np.nan
        eval_eer = (float(eval_raw) * 100.0) if eval_raw is not None else np.nan
        dev_eers.append(dev_eer)
        eval_eers.append(eval_eer)
        gaps.append(eval_eer - dev_eer if np.isfinite(dev_eer) and np.isfinite(eval_eer) else np.nan)
    
    # Plot bars
    bars_dev = ax.bar(
        x - width/2, dev_eers, width,
        label='Dev (In-Domain)',
        color=COLORS["dev"],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
    )
    
    bars_eval = ax.bar(
        x + width/2, eval_eers, width,
        label='Eval (OOD)',
        color=COLORS["eval"],
        edgecolor='black',
        linewidth=0.5,
        alpha=0.85,
    )
    
    # Add gap arrows and annotations where both dev+eval exist
    for i, (dev_bar, eval_bar, gap) in enumerate(zip(bars_dev, bars_eval, gaps)):
        if not np.isfinite(gap):
            continue
        dev_height = dev_bar.get_height()
        eval_height = eval_bar.get_height()
        
        # Arrow from dev to eval
        arrow_x = x[i]
        ax.annotate(
            '',
            xy=(arrow_x + width/2, eval_height),
            xytext=(arrow_x - width/2, dev_height),
            arrowprops=dict(
                arrowstyle='->',
                color=COLORS["gap_arrow"],
                lw=1.5,
                connectionstyle='arc3,rad=0.2',
            ),
        )
        
        # Gap label
        gap_y = max(dev_height, eval_height) + 0.5
        ax.annotate(
            f'+{gap:.1f}%',
            xy=(arrow_x, gap_y),
            ha='center', va='bottom',
            fontsize=10,
            fontweight='bold',
            color=COLORS["gap_arrow"],
        )
    
    # Add reduction callouts for the canonical ERM->DANN pairs when both gaps exist.
    pair_specs = [("wavlm_erm", "wavlm_dann", "WavLM"), ("w2v2_erm", "w2v2_dann", "Wav2Vec2")]
    key_to_index = {key: idx for idx, (_, key) in enumerate(groups)}
    finite_eval_eers = [value for value in eval_eers if np.isfinite(value)]
    max_y = (max(finite_eval_eers) + 3.0) if finite_eval_eers else 3.0
    for erm_key, dann_key, label in pair_specs:
        if erm_key not in key_to_index or dann_key not in key_to_index:
            continue
        erm_gap = gaps[key_to_index[erm_key]]
        dann_gap = gaps[key_to_index[dann_key]]
        if not np.isfinite(erm_gap) or not np.isfinite(dann_gap) or erm_gap <= 0:
            continue
        reduction = (erm_gap - dann_gap) / erm_gap * 100.0
        mid_x = (key_to_index[erm_key] + key_to_index[dann_key]) / 2.0
        ax.annotate(
            f'{label} Gap ↓{reduction:.1f}%',
            xy=(mid_x, max_y + 1),
            ha='center', va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.8),
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('EER (%)')
    ax.set_title('OOD Generalization Gap: Dev (In-Domain) vs Eval (Out-of-Domain)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], rotation=20, ha='right')
    
    # Y-axis limits
    finite_values = [value for value in dev_eers + eval_eers if np.isfinite(value)]
    y_max = max(finite_values) + 6 if finite_values else 10
    ax.set_ylim(0, y_max)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    return fig


def plot_ood_gap_slope(
    data: Dict[str, Any],
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create slope chart showing dev→eval transition with gap annotations.
    
    Alternative visualization showing the "slope" from dev to eval for each model.
    """
    plt.rcParams.update(STYLE_CONFIG)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
    color_cycle = ["#E57373", "#64B5F6", "#FFB74D", "#81C784", "#BA68C8", "#A1887F", "#4DB6AC", "#9575CD"]
    available_models = [model_key for model_key in MODEL_ORDER if model_key in data]
    if not available_models:
        raise ValueError("No model data available for OOD gap slope")
    
    x_positions = [0, 1]  # Dev (0) and Eval (1)
    
    for idx, model_key in enumerate(available_models):
        props = {
            "label": MODEL_LABELS.get(model_key, model_key),
            "color": color_cycle[idx % len(color_cycle)],
            "marker": marker_cycle[idx % len(marker_cycle)],
            "linestyle": "-" if "erm" in model_key else "--",
        }
        model_data = data.get(model_key, {})
        dev_raw = model_data.get("dev_eer")
        eval_raw = model_data.get("eval_eer")
        if dev_raw is None or eval_raw is None:
            continue
        dev_eer = float(dev_raw) * 100
        eval_eer = float(eval_raw) * 100
        
        ax.plot(
            x_positions, [dev_eer, eval_eer],
            marker=props["marker"],
            markersize=10,
            linewidth=2.5,
            color=props["color"],
            linestyle=props["linestyle"],
            label=props["label"],
        )
        
        # Add gap annotation at the eval point
        gap = eval_eer - dev_eer
        ax.annotate(
            f'+{gap:.1f}%',
            xy=(1.05, eval_eer),
            ha='left', va='center',
            fontsize=9,
            color=props["color"],
        )
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('EER (%)')
    ax.set_title('OOD Generalization: Dev → Eval Transition')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Dev\n(In-Domain)', 'Eval\n(Out-of-Domain)'], fontsize=11)
    ax.set_xlim(-0.2, 1.4)
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate OOD gap visualization for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Runs directory containing model subdirectories",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override path to main results JSON",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("figures/ood_gap.png"),
        help="Output file path (PNG and PDF will be saved)",
    )
    p.add_argument(
        "--style",
        choices=["bars", "slope", "both"],
        default="bars",
        help="Visualization style: bars (grouped), slope (line), or both",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data instead of loading from file",
    )
    p.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 10 6)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats (default: 300)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load or generate data
    if args.demo:
        data = generate_demo_data()
    elif args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            logger.info("Use --demo or omit --input to load from --results-dir")
            return 1
        data = load_main_results(args.input)
    else:
        data = load_main_results_from_runs(args.results_dir)
        if not data:
            logger.error(f"No run-derived metrics found under: {args.results_dir}")
            logger.info("Use --demo for synthetic data or --input for JSON override")
            return 1
    
    # Output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    if args.style in ["bars", "both"]:
        fig = plot_ood_gap_bars(data, figsize=tuple(args.figsize))
        
        # Save PNG
        png_path = args.output.with_suffix('.png')
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG: {png_path}")
        
        # Save PDF
        pdf_path = args.output.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
    
    if args.style in ["slope", "both"]:
        fig = plot_ood_gap_slope(data, figsize=tuple(args.figsize))
        
        suffix = "_slope" if args.style == "both" else ""
        
        # Save PNG
        png_path = args.output.with_stem(args.output.stem + suffix).with_suffix('.png')
        fig.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG: {png_path}")
        
        # Save PDF
        pdf_path = args.output.with_stem(args.output.stem + suffix).with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF: {pdf_path}")
        
        plt.close(fig)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("OOD Gap Figure Summary")
    logger.info("=" * 60)
    
    for model_key in [key for key in MODEL_ORDER if key in data]:
        model_data = data.get(model_key, {})
        dev_eer = model_data.get("dev_eer")
        eval_eer = model_data.get("eval_eer")
        if dev_eer is None or eval_eer is None:
            logger.info(f"{model_key}: Dev=-, Eval={eval_eer * 100:.2f}% (gap unavailable)" if eval_eer is not None else f"{model_key}: Dev=-, Eval=-, gap unavailable")
            continue
        gap = (eval_eer - dev_eer) * 100
        logger.info(f"{model_key}: Dev={dev_eer*100:.2f}%, Eval={eval_eer*100:.2f}%, Gap={gap:.2f}%")
    
    # Calculate reductions
    wavlm_erm = data.get("wavlm_erm", {})
    wavlm_dann = data.get("wavlm_dann", {})
    wavlm_erm_gap = (
        wavlm_erm["eval_eer"] - wavlm_erm["dev_eer"]
        if wavlm_erm.get("eval_eer") is not None and wavlm_erm.get("dev_eer") is not None
        else None
    )
    wavlm_dann_gap = (
        wavlm_dann["eval_eer"] - wavlm_dann["dev_eer"]
        if wavlm_dann.get("eval_eer") is not None and wavlm_dann.get("dev_eer") is not None
        else None
    )

    if wavlm_erm_gap is not None and wavlm_dann_gap is not None and wavlm_erm_gap > 0:
        wavlm_reduction = (wavlm_erm_gap - wavlm_dann_gap) / wavlm_erm_gap * 100
        logger.info(f"\nWavLM Gap Reduction: {wavlm_reduction:.1f}%")
    
    logger.info(f"\nOutput: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
