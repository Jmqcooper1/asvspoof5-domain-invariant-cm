"""Core evaluation metrics for ASVspoof 5.

Implements:
- EER (Equal Error Rate)
- minDCF (Minimum Detection Cost Function)
- Cllr (Log-Likelihood Ratio Cost)
- actDCF (Actual DCF at fixed threshold)
"""

import numpy as np
from typing import Optional


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate.

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).

    Returns:
        Tuple of (EER, threshold at EER).
    """
    # Sort scores ascending
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Count classes (0 = bonafide, 1 = spoof)
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 0.5, 0.0

    # At each threshold position i, samples 0..i are below threshold (rejected as spoof)
    # FRR = fraction of bonafide incorrectly rejected (bonafide below threshold)
    # FAR = fraction of spoof incorrectly accepted (spoof above threshold)
    bonafide_below = np.cumsum(sorted_labels == 0)
    frr = bonafide_below / n_bonafide

    spoof_below = np.cumsum(sorted_labels == 1)
    spoof_above = n_spoof - spoof_below
    far = spoof_above / n_spoof

    # Find EER (where FRR = FAR)
    diff = frr - far
    idx = np.argmin(np.abs(diff))

    eer = (frr[idx] + far[idx]) / 2
    threshold = sorted_scores[idx]

    return float(eer), float(threshold)


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05,
) -> float:
    """Compute minimum Detection Cost Function.

    DCF = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        c_miss: Cost of miss (false rejection of bonafide).
        c_fa: Cost of false alarm (false acceptance of spoof).
        p_target: Prior probability of target (bonafide).

    Returns:
        Minimum DCF value.
    """
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 1.0

    # Sort scores ascending
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # At each threshold, samples below are rejected (classified as spoof)
    # p_miss = P(rejected | bonafide) = bonafide below threshold / total bonafide
    # p_fa = P(accepted | spoof) = spoof above threshold / total spoof
    bonafide_below = np.cumsum(sorted_labels == 0)
    p_miss = bonafide_below / n_bonafide

    spoof_below = np.cumsum(sorted_labels == 1)
    spoof_above = n_spoof - spoof_below
    p_fa = spoof_above / n_spoof

    # Compute DCF at each threshold
    dcf = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    # Normalize by minimum of default DCFs
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = np.min(dcf) / default_dcf

    return float(min_dcf)


def compute_cllr(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute Log-Likelihood Ratio Cost (Cllr).

    Measures calibration quality of scores as likelihood ratios.

    Args:
        scores: Log-likelihood ratio scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).

    Returns:
        Cllr value.
    """
    # Separate scores by class (0 = bonafide, 1 = spoof)
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 1.0

    # Compute Cllr components
    # For bonafide: average of log2(1 + e^(-score)) - penalizes low scores
    # For spoof: average of log2(1 + e^(score)) - penalizes high scores

    def softplus_log2(x):
        # log2(1 + e^x) = x / log(2) + log2(1 + e^(-x))
        # Numerically stable version
        return np.log2(1 + np.exp(-np.abs(x))) + np.maximum(x, 0) / np.log(2)

    cllr_bonafide = np.mean(softplus_log2(-bonafide_scores))
    cllr_spoof = np.mean(softplus_log2(spoof_scores))

    cllr = (cllr_bonafide + cllr_spoof) / 2

    return float(cllr)


def compute_act_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05,
) -> float:
    """Compute actual DCF at a fixed threshold.

    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
        threshold: Decision threshold (score >= threshold -> accept as bonafide).
        c_miss: Cost of miss (false rejection of bonafide).
        c_fa: Cost of false alarm (false acceptance of spoof).
        p_target: Prior probability of target (bonafide).

    Returns:
        Actual DCF value.
    """
    # score >= threshold means accepted as bonafide
    accepted = scores >= threshold

    # Count classes (0 = bonafide, 1 = spoof)
    n_bonafide = np.sum(labels == 0)
    n_spoof = np.sum(labels == 1)

    if n_bonafide == 0 or n_spoof == 0:
        return 1.0

    # p_miss = P(rejected | bonafide) = bonafide not accepted / total bonafide
    p_miss = np.sum((~accepted) & (labels == 0)) / n_bonafide

    # p_fa = P(accepted | spoof) = spoof accepted / total spoof
    p_fa = np.sum(accepted & (labels == 1)) / n_spoof

    # Compute DCF
    dcf = c_miss * p_target * p_miss + c_fa * (1 - p_target) * p_fa

    # Normalize
    default_dcf = min(c_miss * p_target, c_fa * (1 - p_target))
    act_dcf = dcf / default_dcf

    return float(act_dcf)


def bootstrap_metric(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Compute metric with bootstrap confidence interval.

    Args:
        scores: Detection scores.
        labels: Binary labels.
        metric_fn: Function that takes (scores, labels) and returns metric.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (mean, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    n_samples = len(scores)

    bootstrap_values = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        value = metric_fn(scores[indices], labels[indices])
        bootstrap_values.append(value)

    bootstrap_values = np.array(bootstrap_values)
    mean = np.mean(bootstrap_values)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_values, alpha / 2 * 100)
    upper = np.percentile(bootstrap_values, (1 - alpha / 2) * 100)

    return float(mean), float(lower), float(upper)
