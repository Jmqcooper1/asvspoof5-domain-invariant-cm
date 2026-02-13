#!/usr/bin/env python3
"""Generate LaTeX and Markdown tables for thesis.

This script generates all tables specified in docs/thesis_figures_spec.md:
- T1: Main results table (Dev/Eval EER, minDCF)
- T2: Per-codec EER comparison
- T3: OOD gap analysis
- T4: Projection probe results
- T5: Dataset statistics
- T6: Synthetic augmentation coverage
- T7: Hyperparameters

Usage:
    # Generate all tables with default paths
    python scripts/generate_thesis_tables.py

    # Generate with custom paths
    python scripts/generate_thesis_tables.py \\
        --main-results results/main_results.json \\
        --per-codec results/per_codec_eer.json \\
        --projection-wavlm results/rq3_projection.json \\
        --projection-w2v2 results/rq3_projection_w2v2.json \\
        --output-dir figures/tables \\
        --verbose

    # Generate only specific tables
    python scripts/generate_thesis_tables.py --tables T1 T4 T5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading Utilities
# ---------------------------------------------------------------------------
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def safe_get(data: Optional[Dict], *keys, default=None):
    """Safely get nested dictionary value."""
    if data is None:
        return default
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


# ---------------------------------------------------------------------------
# Table Generation: LaTeX and Markdown
# ---------------------------------------------------------------------------
def to_latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    column_format: Optional[str] = None,
) -> str:
    """Convert headers and rows to LaTeX table format."""
    if column_format is None:
        column_format = "l" + "c" * (len(headers) - 1)
    
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Convert headers and rows to Markdown table format."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_table(
    output_dir: Path,
    name: str,
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
) -> None:
    """Save table in both LaTeX and Markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LaTeX
    tex_path = output_dir / f"{name}.tex"
    tex_content = to_latex_table(headers, rows, caption, label)
    tex_path.write_text(tex_content)
    logger.info(f"Saved LaTeX: {tex_path}")
    
    # Markdown
    md_path = output_dir / f"{name}.md"
    md_content = f"# {caption}\n\n" + to_markdown_table(headers, rows)
    md_path.write_text(md_content)
    logger.info(f"Saved Markdown: {md_path}")


# ---------------------------------------------------------------------------
# T1: Main Results Table
# ---------------------------------------------------------------------------
def generate_t1_main_results(
    main_results: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T1: Main results table."""
    logger.info("Generating T1: Main Results Table")
    
    headers = ["Model", "Backbone", "Dev EER (%)", "Eval EER (%)", "Eval minDCF"]
    
    # Default/placeholder data if no results file
    if main_results is None:
        rows = [
            ["ERM", "WavLM", "—", "—", "—"],
            ["DANN", "WavLM", "—", "—", "—"],
            ["ERM", "W2V2", "—", "—", "—"],
            ["DANN", "W2V2", "—", "—", "—"],
        ]
        logger.warning("Using placeholder data for T1 (no main_results.json)")
    else:
        rows = []
        for model_key in ["wavlm_erm", "wavlm_dann", "w2v2_erm", "w2v2_dann"]:
            model_data = main_results.get(model_key, {})
            backbone = "WavLM" if "wavlm" in model_key else "W2V2"
            method = "DANN" if "dann" in model_key else "ERM"
            
            dev_eer = model_data.get("dev_eer", "—")
            eval_eer = model_data.get("eval_eer", "—")
            eval_mindcf = model_data.get("eval_mindcf", "—")
            
            if isinstance(dev_eer, float):
                dev_eer = f"{dev_eer * 100:.2f}"
            if isinstance(eval_eer, float):
                eval_eer = f"{eval_eer * 100:.2f}"
            if isinstance(eval_mindcf, float):
                eval_mindcf = f"{eval_mindcf:.4f}"
            
            rows.append([method, backbone, str(dev_eer), str(eval_eer), str(eval_mindcf)])
    
    save_table(
        output_dir, "T1_main_results", headers, rows,
        caption="Main Results: EER and minDCF for ERM vs DANN",
        label="tab:main_results",
    )
    return True


# ---------------------------------------------------------------------------
# T2: Per-Codec EER Comparison
# ---------------------------------------------------------------------------
def generate_t2_per_codec(
    per_codec: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T2: Per-codec EER comparison table."""
    logger.info("Generating T2: Per-Codec EER Comparison")
    
    headers = ["Codec", "WavLM ERM", "WavLM DANN", "W2V2 ERM", "W2V2 DANN"]
    
    # Default codec list
    codecs = ["C01", "C02", "C03", "C04", "C05", "C06", 
              "C07", "C08", "C09", "C10", "C11", "NONE"]
    
    if per_codec is None:
        rows = [[codec, "—", "—", "—", "—"] for codec in codecs]
        logger.warning("Using placeholder data for T2 (no per_codec_eer.json)")
    else:
        rows = []
        for codec in codecs:
            codec_data = per_codec.get(codec, {})
            row = [codec]
            for model in ["wavlm_erm", "wavlm_dann", "w2v2_erm", "w2v2_dann"]:
                eer = codec_data.get(model, "—")
                if isinstance(eer, float):
                    eer = f"{eer * 100:.2f}"
                row.append(str(eer))
            rows.append(row)
    
    save_table(
        output_dir, "T2_per_codec", headers, rows,
        caption="Per-Codec EER (\\%) on Eval Set",
        label="tab:per_codec",
    )
    return True


# ---------------------------------------------------------------------------
# T3: OOD Gap Analysis
# ---------------------------------------------------------------------------
def generate_t3_ood_gap(
    main_results: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T3: OOD gap analysis table."""
    logger.info("Generating T3: OOD Gap Analysis")
    
    headers = ["Model", "Backbone", "Dev EER (%)", "Eval EER (%)", "Gap", "Gap Reduction"]
    
    if main_results is None:
        rows = [
            ["ERM", "WavLM", "—", "—", "—", "—"],
            ["DANN", "WavLM", "—", "—", "—", "—"],
            ["ERM", "W2V2", "—", "—", "—", "—"],
            ["DANN", "W2V2", "—", "—", "—", "—"],
        ]
        logger.warning("Using placeholder data for T3 (no main_results.json)")
    else:
        rows = []
        # Calculate baseline gaps for each backbone
        for backbone in ["wavlm", "w2v2"]:
            backbone_name = "WavLM" if backbone == "wavlm" else "W2V2"
            erm_data = main_results.get(f"{backbone}_erm", {})
            dann_data = main_results.get(f"{backbone}_dann", {})
            
            erm_dev = erm_data.get("dev_eer", 0)
            erm_eval = erm_data.get("eval_eer", 0)
            dann_dev = dann_data.get("dev_eer", 0)
            dann_eval = dann_data.get("eval_eer", 0)
            
            erm_gap = (erm_eval - erm_dev) if isinstance(erm_dev, float) and isinstance(erm_eval, float) else None
            dann_gap = (dann_eval - dann_dev) if isinstance(dann_dev, float) and isinstance(dann_eval, float) else None
            
            # ERM row
            erm_dev_str = f"{erm_dev * 100:.2f}" if isinstance(erm_dev, float) else "—"
            erm_eval_str = f"{erm_eval * 100:.2f}" if isinstance(erm_eval, float) else "—"
            erm_gap_str = f"{erm_gap * 100:.2f}" if erm_gap is not None else "—"
            rows.append(["ERM", backbone_name, erm_dev_str, erm_eval_str, erm_gap_str, "(baseline)"])
            
            # DANN row
            dann_dev_str = f"{dann_dev * 100:.2f}" if isinstance(dann_dev, float) else "—"
            dann_eval_str = f"{dann_eval * 100:.2f}" if isinstance(dann_eval, float) else "—"
            dann_gap_str = f"{dann_gap * 100:.2f}" if dann_gap is not None else "—"
            
            if erm_gap is not None and dann_gap is not None and erm_gap > 0:
                reduction = ((erm_gap - dann_gap) / erm_gap) * 100
                reduction_str = f"{reduction:.1f}\\%"
            else:
                reduction_str = "—"
            
            rows.append(["DANN", backbone_name, dann_dev_str, dann_eval_str, dann_gap_str, reduction_str])
    
    save_table(
        output_dir, "T3_ood_gap", headers, rows,
        caption="OOD Gap Analysis: Dev vs Eval Generalization",
        label="tab:ood_gap",
    )
    return True


# ---------------------------------------------------------------------------
# T4: Projection Probe Results
# ---------------------------------------------------------------------------
def generate_t4_projection_probes(
    projection_wavlm: Optional[Dict[str, Any]],
    projection_w2v2: Optional[Dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Generate T4: Projection probe results table."""
    logger.info("Generating T4: Projection Probe Results")
    
    headers = ["Backbone", "ERM Probe Acc", "DANN Probe Acc", "Reduction", "Rel. Reduction"]
    rows = []
    
    for backbone_name, proj_data in [("WavLM", projection_wavlm), ("W2V2", projection_w2v2)]:
        if proj_data is None:
            rows.append([backbone_name, "—", "—", "—", "—"])
            continue
        
        erm_acc = safe_get(proj_data, "results", "erm", "codec", "accuracy", default=None)
        dann_acc = safe_get(proj_data, "results", "dann", "codec", "accuracy", default=None)
        reduction = safe_get(proj_data, "comparison", "codec", "reduction", default=None)
        rel_reduction = safe_get(proj_data, "comparison", "codec", "relative_reduction", default=None)
        
        erm_str = f"{erm_acc:.3f}" if erm_acc is not None else "—"
        dann_str = f"{dann_acc:.3f}" if dann_acc is not None else "—"
        red_str = f"{reduction:.3f}" if reduction is not None else "—"
        rel_str = f"{rel_reduction * 100:.1f}\\%" if rel_reduction is not None else "—"
        
        rows.append([backbone_name, erm_str, dann_str, red_str, rel_str])
    
    save_table(
        output_dir, "T4_projection_probes", headers, rows,
        caption="Projection Layer Codec Probe Accuracy (RQ3)",
        label="tab:projection_probes",
    )
    return True


# ---------------------------------------------------------------------------
# T5: Dataset Statistics
# ---------------------------------------------------------------------------
def generate_t5_dataset_stats(output_dir: Path) -> bool:
    """Generate T5: Dataset statistics table."""
    logger.info("Generating T5: Dataset Statistics")
    
    headers = ["Split", "Bonafide", "Spoof", "Total", "Codecs", "Duration (h)"]
    
    # These are the official ASVspoof 5 Track 1 statistics
    rows = [
        ["Train", "18,797", "163,560", "182,357", "—", "~144"],
        ["Dev", "31,336", "108,978", "140,314", "—", "~100"],
        ["Eval", "133,320", "542,751", "676,071", "12", "~550"],
    ]
    
    save_table(
        output_dir, "T5_dataset_stats", headers, rows,
        caption="ASVspoof 5 Track 1 Dataset Statistics",
        label="tab:dataset_stats",
    )
    return True


# ---------------------------------------------------------------------------
# T6: Synthetic Augmentation Coverage
# ---------------------------------------------------------------------------
def generate_t6_augmentation(output_dir: Path) -> bool:
    """Generate T6: Synthetic augmentation coverage table."""
    logger.info("Generating T6: Synthetic Augmentation Coverage")
    
    headers = ["Codec", "Quality Levels", "Bitrates"]
    
    # Synthetic augmentation config
    rows = [
        ["MP3 (libmp3lame)", "4", "64k, 96k, 128k, 192k"],
        ["AAC (aac)", "4", "64k, 96k, 128k, 192k"],
        ["Opus (libopus)", "4", "32k, 64k, 96k, 128k"],
    ]
    
    save_table(
        output_dir, "T6_augmentation", headers, rows,
        caption="Synthetic Codec Augmentation for DANN Training",
        label="tab:augmentation",
    )
    return True


# ---------------------------------------------------------------------------
# T7: Hyperparameters
# ---------------------------------------------------------------------------
def generate_t7_hyperparameters(output_dir: Path) -> bool:
    """Generate T7: Hyperparameters table."""
    logger.info("Generating T7: Hyperparameters")
    
    headers = ["Category", "Parameter", "Value"]
    
    rows = [
        # Model architecture
        ["Architecture", "Backbone", "WavLM Base+ / W2V2 Base"],
        ["", "Backbone layers", "12 (frozen)"],
        ["", "Projection dim", "256"],
        ["", "Classifier hidden", "256"],
        ["", "Domain head hidden", "256"],
        # Training
        ["Training", "Batch size", "32"],
        ["", "Learning rate", "1e-4"],
        ["", "Optimizer", "AdamW"],
        ["", "Weight decay", "0.01"],
        ["", "Epochs", "10"],
        ["", "Early stopping", "Patience 3"],
        # DANN-specific
        ["DANN", r"$\\lambda$ schedule", "Linear ramp 0→1"],
        ["", "GRL", "Gradient reversal layer"],
        ["", "Domain targets", "CODEC + CODEC\\_Q"],
        # Regularization
        ["Regularization", "Dropout", "0.1"],
        ["", "Label smoothing", "0.0"],
    ]
    
    save_table(
        output_dir, "T7_hyperparameters", headers, rows,
        caption="Training Hyperparameters",
        label="tab:hyperparameters",
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LaTeX and Markdown tables for thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input files
    p.add_argument(
        "--main-results",
        type=Path,
        default=Path("results/main_results.json"),
        help="Path to main results JSON",
    )
    p.add_argument(
        "--per-codec",
        type=Path,
        default=Path("results/per_codec_eer.json"),
        help="Path to per-codec EER JSON",
    )
    p.add_argument(
        "--projection-wavlm",
        type=Path,
        default=Path("results/rq3_projection.json"),
        help="Path to WavLM projection probes JSON",
    )
    p.add_argument(
        "--projection-w2v2",
        type=Path,
        default=Path("results/rq3_projection_w2v2.json"),
        help="Path to W2V2 projection probes JSON",
    )
    
    # Output
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/tables"),
        help="Output directory for tables",
    )
    
    # Table selection
    p.add_argument(
        "--tables",
        nargs="+",
        choices=["T1", "T2", "T3", "T4", "T5", "T6", "T7", "all"],
        default=["all"],
        help="Which tables to generate (default: all)",
    )
    
    # Verbosity
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
    
    # Load available data
    main_results = load_json(args.main_results)
    per_codec = load_json(args.per_codec)
    projection_wavlm = load_json(args.projection_wavlm)
    projection_w2v2 = load_json(args.projection_w2v2)
    
    # Determine which tables to generate
    tables = args.tables
    if "all" in tables:
        tables = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
    
    # Generate tables
    output_dir = args.output_dir
    success_count = 0
    
    if "T1" in tables:
        if generate_t1_main_results(main_results, output_dir):
            success_count += 1
    
    if "T2" in tables:
        if generate_t2_per_codec(per_codec, output_dir):
            success_count += 1
    
    if "T3" in tables:
        if generate_t3_ood_gap(main_results, output_dir):
            success_count += 1
    
    if "T4" in tables:
        if generate_t4_projection_probes(projection_wavlm, projection_w2v2, output_dir):
            success_count += 1
    
    if "T5" in tables:
        if generate_t5_dataset_stats(output_dir):
            success_count += 1
    
    if "T6" in tables:
        if generate_t6_augmentation(output_dir):
            success_count += 1
    
    if "T7" in tables:
        if generate_t7_hyperparameters(output_dir):
            success_count += 1
    
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Generated {success_count}/{len(tables)} tables")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
