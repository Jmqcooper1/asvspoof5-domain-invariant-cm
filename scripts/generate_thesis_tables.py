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
import csv
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

# Matches notebook naming for shared models.
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
PRIMARY_MODEL_KEYS = ["wavlm_erm", "wavlm_dann", "w2v2_erm", "w2v2_dann"]
CODEC_ORDER = ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "NONE"]
GAP_BASELINE_MODEL = {"wavlm_dann": "wavlm_erm", "w2v2_dann": "w2v2_erm", "w2v2_dann_v2": "w2v2_erm"}
MODEL_METADATA = {
    "wavlm_erm": ("ERM", "WavLM"),
    "wavlm_dann": ("DANN", "WavLM"),
    "w2v2_erm": ("ERM", "Wav2Vec2"),
    "w2v2_dann": ("DANN v1", "Wav2Vec2"),
    "w2v2_dann_v2": ("DANN v2", "Wav2Vec2"),
    "lfcc_gmm": ("GMM", "LFCC"),
    "trillsson_logistic": ("Logistic", "TRILLsson"),
    "trillsson_mlp": ("MLP", "TRILLsson"),
}


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


def get_nested_float(data: Dict[str, Any], *keys: str) -> Optional[float]:
    value = safe_get(data, *keys, default=None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_first_numeric_value(data: Dict[str, Any], keys: List[str]) -> Optional[float]:
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
    final_val_eer = get_nested_float(payload, "final_val", "eer")
    if final_val_eer is not None:
        return final_val_eer
    return get_first_numeric_value(payload, ["best_eer"])


def load_eval_results_from_runs(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load eval metrics from results/runs/*/eval_eval/metrics.json."""
    results: Dict[str, Dict[str, Any]] = {}
    for model_key in MODEL_ORDER:
        run_dir = MODEL_RUN_DIR[model_key]
        metrics_path = results_dir / run_dir / "eval_eval" / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = load_json(metrics_path)
        if payload is None:
            continue
        results[model_key] = {
            "eval_eer": get_first_numeric_value(payload, ["eer", "eval_eer"]),
            "eval_mindcf": get_first_numeric_value(payload, ["min_dcf", "eval_mindcf"]),
        }
    return results


def load_dev_eer_from_runs(results_dir: Path) -> Dict[str, float]:
    """Load dev EER with fallback chain.

    1) eval_dev/metrics.json
    2) metrics.json
    3) metrics_train.json
    """
    dev_values: Dict[str, float] = {}
    for model_key in MODEL_ORDER:
        run_dir = MODEL_RUN_DIR[model_key]
        candidate_paths = [
            results_dir / run_dir / "eval_dev" / "metrics.json",
            results_dir / run_dir / "metrics.json",
            results_dir / run_dir / "metrics_train.json",
        ]
        for path in candidate_paths:
            if not path.exists():
                continue
            payload = load_json(path)
            if payload is None:
                continue
            dev_eer = extract_dev_eer_from_payload(payload)
            if dev_eer is not None:
                dev_values[model_key] = dev_eer
                break
    return dev_values


def load_per_codec_from_runs(results_dir: Path) -> Dict[str, Dict[str, float]]:
    """Load codec-wise EER values from metrics_by_codec.csv for all models."""
    per_codec: Dict[str, Dict[str, float]] = {}
    for model_key in MODEL_ORDER:
        run_dir = MODEL_RUN_DIR[model_key]
        csv_path = results_dir / run_dir / "eval_eval" / "tables" / "metrics_by_codec.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                codec = row.get("domain")
                eer_raw = row.get("eer")
                if not codec or eer_raw is None:
                    continue
                try:
                    eer = float(eer_raw)
                except (TypeError, ValueError):
                    continue
                per_codec.setdefault(codec, {})[model_key] = eer
    return per_codec


def merge_main_results_from_runs(results_dir: Path) -> Optional[Dict[str, Dict[str, Any]]]:
    eval_results = load_eval_results_from_runs(results_dir)
    dev_values = load_dev_eer_from_runs(results_dir)
    if not eval_results and not dev_values:
        return None
    merged: Dict[str, Dict[str, Any]] = {}
    for model_key in MODEL_ORDER:
        if model_key not in eval_results and model_key not in dev_values:
            continue
        merged[model_key] = {}
        if model_key in eval_results:
            merged[model_key].update(eval_results[model_key])
        if model_key in dev_values:
            merged[model_key]["dev_eer"] = dev_values[model_key]
    return merged


def format_optional_percent(value: Optional[Any]) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return "-"


def format_optional_decimal(value: Optional[Any], precision: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "-"


def get_ordered_available_models(
    data: Optional[Dict[str, Any]],
    default_models: Optional[List[str]] = None,
) -> List[str]:
    if default_models is None:
        default_models = PRIMARY_MODEL_KEYS
    if data is None:
        return default_models
    available = [model for model in MODEL_ORDER if model in data]
    return available or default_models


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
    
    if main_results is None:
        rows = [
            ["ERM", "WavLM", "-", "-", "-"],
            ["DANN", "WavLM", "-", "-", "-"],
            ["ERM", "Wav2Vec2", "-", "-", "-"],
            ["DANN", "Wav2Vec2", "-", "-", "-"],
        ]
        logger.warning("Using placeholder data for T1 (no loaded main results)")
    else:
        rows = []
        for model_key in get_ordered_available_models(main_results):
            model_data = main_results.get(model_key, {})
            method, backbone = MODEL_METADATA.get(
                model_key,
                (MODEL_LABELS.get(model_key, model_key), "Unknown"),
            )
            dev_eer = format_optional_percent(model_data.get("dev_eer"))
            eval_eer = format_optional_percent(model_data.get("eval_eer"))
            eval_mindcf = format_optional_decimal(model_data.get("eval_mindcf"), precision=4)
            rows.append([method, backbone, dev_eer, eval_eer, eval_mindcf])
    
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
    
    models = PRIMARY_MODEL_KEYS
    if per_codec:
        discovered_models = {model for codec_values in per_codec.values() for model in codec_values}
        models = [model for model in MODEL_ORDER if model in discovered_models]
        if not models:
            models = PRIMARY_MODEL_KEYS

    headers = ["Codec"] + [MODEL_LABELS.get(model, model) for model in models]

    codecs = [codec for codec in CODEC_ORDER if per_codec is None or codec in per_codec]
    if per_codec:
        extras = sorted(codec for codec in per_codec if codec not in CODEC_ORDER)
        codecs.extend(extras)
    if not codecs:
        codecs = CODEC_ORDER

    if per_codec is None:
        rows = [[codec] + ["-"] * len(models) for codec in codecs]
        logger.warning("Using placeholder data for T2 (no loaded per-codec data)")
    else:
        rows = []
        for codec in codecs:
            codec_data = per_codec.get(codec, {})
            row = [codec]
            for model in models:
                row.append(format_optional_percent(codec_data.get(model)))
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
            ["ERM", "WavLM", "-", "-", "-", "-"],
            ["DANN", "WavLM", "-", "-", "-", "-"],
            ["ERM", "Wav2Vec2", "-", "-", "-", "-"],
            ["DANN", "Wav2Vec2", "-", "-", "-", "-"],
        ]
        logger.warning("Using placeholder data for T3 (no loaded main results)")
    else:
        rows = []
        available_models = get_ordered_available_models(main_results)
        gap_by_model: Dict[str, Optional[float]] = {}

        for model_key in available_models:
            model_data = main_results.get(model_key, {})
            dev_eer = model_data.get("dev_eer")
            eval_eer = model_data.get("eval_eer")
            if dev_eer is None or eval_eer is None:
                gap_by_model[model_key] = None
                continue
            try:
                gap_by_model[model_key] = float(eval_eer) - float(dev_eer)
            except (TypeError, ValueError):
                gap_by_model[model_key] = None

        for model_key in available_models:
            model_data = main_results.get(model_key, {})
            method, backbone = MODEL_METADATA.get(
                model_key,
                (MODEL_LABELS.get(model_key, model_key), "Unknown"),
            )
            gap_value = gap_by_model.get(model_key)
            if model_key in GAP_BASELINE_MODEL:
                baseline_key = GAP_BASELINE_MODEL[model_key]
                baseline_gap = gap_by_model.get(baseline_key)
                if baseline_gap is not None and gap_value is not None and baseline_gap > 0:
                    reduction = (baseline_gap - gap_value) / baseline_gap * 100.0
                    reduction_str = f"{reduction:.1f}\\%"
                else:
                    reduction_str = "-"
            elif model_key.endswith("_erm"):
                reduction_str = "(baseline)"
            else:
                reduction_str = "-"

            rows.append(
                [
                    method,
                    backbone,
                    format_optional_percent(model_data.get("dev_eer")),
                    format_optional_percent(model_data.get("eval_eer")),
                    format_optional_percent(gap_value),
                    reduction_str,
                ]
            )
    
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
            rows.append([backbone_name, "-", "-", "-", "-"])
            continue
        
        erm_acc = safe_get(proj_data, "results", "erm", "codec", "accuracy", default=None)
        dann_acc = safe_get(proj_data, "results", "dann", "codec", "accuracy", default=None)
        reduction = safe_get(proj_data, "comparison", "codec", "reduction", default=None)
        rel_reduction = safe_get(proj_data, "comparison", "codec", "relative_reduction", default=None)
        
        erm_str = f"{erm_acc:.3f}" if erm_acc is not None else "-"
        dann_str = f"{dann_acc:.3f}" if dann_acc is not None else "-"
        red_str = f"{reduction:.3f}" if reduction is not None else "-"
        rel_str = f"{rel_reduction * 100:.1f}\\%" if rel_reduction is not None else "-"
        
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
    
    # Input paths and optional JSON overrides
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/runs"),
        help="Runs directory containing model subdirectories",
    )
    p.add_argument(
        "--main-results",
        type=Path,
        default=None,
        help="Override path to aggregated main results JSON",
    )
    p.add_argument(
        "--per-codec",
        type=Path,
        default=None,
        help="Override path to aggregated per-codec EER JSON",
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
    
    # Load available data (JSON overrides win; otherwise derive from results/runs)
    if args.main_results:
        main_results = load_json(args.main_results)
    else:
        main_results = merge_main_results_from_runs(args.results_dir)
        if main_results is None:
            logger.warning(f"No run-derived main results found under: {args.results_dir}")

    if args.per_codec:
        per_codec = load_json(args.per_codec)
    else:
        run_per_codec = load_per_codec_from_runs(args.results_dir)
        per_codec = run_per_codec if run_per_codec else None
        if per_codec is None:
            logger.warning(f"No run-derived per-codec CSVs found under: {args.results_dir}")
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
