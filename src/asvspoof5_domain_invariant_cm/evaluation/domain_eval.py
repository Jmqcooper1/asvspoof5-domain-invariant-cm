"""Per-domain evaluation and domain gap analysis."""

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compute_eer, compute_min_dcf


def evaluate_per_domain(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    domain_col: str = "codec",
) -> pd.DataFrame:
    """Compute metrics for each domain value.

    Args:
        df: DataFrame with scores, labels, and domain column.
        score_col: Name of score column.
        label_col: Name of label column (1=bonafide, 0=spoof).
        domain_col: Name of domain column.

    Returns:
        DataFrame with per-domain metrics.
    """
    results = []

    for domain_val, group in df.groupby(domain_col):
        scores = group[score_col].values
        labels = group[label_col].values

        eer, eer_threshold = compute_eer(scores, labels)
        min_dcf = compute_min_dcf(scores, labels)

        results.append(
            {
                "domain": domain_val,
                "n_samples": len(group),
                "n_bonafide": int(np.sum(labels == 1)),
                "n_spoof": int(np.sum(labels == 0)),
                "eer": eer,
                "eer_threshold": eer_threshold,
                "min_dcf": min_dcf,
            }
        )

    return pd.DataFrame(results)


def compute_domain_gap(
    in_domain_metrics: dict,
    out_domain_metrics: dict,
) -> dict:
    """Compute performance gap between in-domain and out-of-domain.

    Args:
        in_domain_metrics: Metrics on in-domain data.
        out_domain_metrics: Metrics on out-of-domain data.

    Returns:
        Dictionary with gap metrics.
    """
    return {
        "eer_gap": out_domain_metrics["eer"] - in_domain_metrics["eer"],
        "min_dcf_gap": (
            out_domain_metrics["min_dcf"] - in_domain_metrics["min_dcf"]
        ),
        "eer_ratio": out_domain_metrics["eer"] / max(in_domain_metrics["eer"], 1e-6),
        "min_dcf_ratio": (
            out_domain_metrics["min_dcf"]
            / max(in_domain_metrics["min_dcf"], 1e-6)
        ),
    }


def held_out_domain_evaluation(
    df: pd.DataFrame,
    hold_out_value: str,
    domain_col: str = "codec",
    score_col: str = "score",
    label_col: str = "label",
    group_col: Optional[str] = "codec_seed",
) -> dict:
    """Evaluate with one domain value held out.

    Args:
        df: DataFrame with predictions.
        hold_out_value: Domain value to hold out for testing.
        domain_col: Name of domain column.
        score_col: Name of score column.
        label_col: Name of label column.
        group_col: Column to group by (for leakage prevention).

    Returns:
        Dictionary with in-domain and out-domain metrics.
    """
    # Split by domain
    in_domain = df[df[domain_col] != hold_out_value]
    out_domain = df[df[domain_col] == hold_out_value]

    # Compute metrics
    in_scores = in_domain[score_col].values
    in_labels = in_domain[label_col].values
    in_eer, _ = compute_eer(in_scores, in_labels)
    in_min_dcf = compute_min_dcf(in_scores, in_labels)

    out_scores = out_domain[score_col].values
    out_labels = out_domain[label_col].values
    out_eer, _ = compute_eer(out_scores, out_labels)
    out_min_dcf = compute_min_dcf(out_scores, out_labels)

    return {
        "hold_out_value": hold_out_value,
        "in_domain": {
            "n_samples": len(in_domain),
            "eer": in_eer,
            "min_dcf": in_min_dcf,
        },
        "out_domain": {
            "n_samples": len(out_domain),
            "eer": out_eer,
            "min_dcf": out_min_dcf,
        },
        "gap": compute_domain_gap(
            {"eer": in_eer, "min_dcf": in_min_dcf},
            {"eer": out_eer, "min_dcf": out_min_dcf},
        ),
    }


def aggregate_domain_results(
    per_domain_df: pd.DataFrame,
    weight_by_samples: bool = True,
) -> dict:
    """Aggregate per-domain results.

    Args:
        per_domain_df: DataFrame from evaluate_per_domain.
        weight_by_samples: If True, compute weighted average.

    Returns:
        Dictionary with aggregated metrics.
    """
    if weight_by_samples:
        total_samples = per_domain_df["n_samples"].sum()
        weights = per_domain_df["n_samples"] / total_samples

        avg_eer = (per_domain_df["eer"] * weights).sum()
        avg_min_dcf = (per_domain_df["min_dcf"] * weights).sum()
    else:
        avg_eer = per_domain_df["eer"].mean()
        avg_min_dcf = per_domain_df["min_dcf"].mean()

    return {
        "avg_eer": float(avg_eer),
        "avg_min_dcf": float(avg_min_dcf),
        "std_eer": float(per_domain_df["eer"].std()),
        "std_min_dcf": float(per_domain_df["min_dcf"].std()),
        "max_eer": float(per_domain_df["eer"].max()),
        "min_eer": float(per_domain_df["eer"].min()),
        "n_domains": len(per_domain_df),
    }
