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
        labels: Binary labels (1 = bonafide, 0 = spoof).

    Returns:
        Tuple of (EER, threshold at EER).
    """
    # Sort scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Count positives and negatives
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0

    # Compute FRR and FAR at each threshold
    # FRR = FN / P (fraction of bonafide rejected)
    # FAR = FP / N (fraction of spoof accepted)
    fn = np.cumsum(sorted_labels == 1)  # False negatives
    tp = n_pos - fn
    frr = fn / n_pos

    fp = np.cumsum(sorted_labels == 0)  # False positives
    tn = n_neg - fp
    far = 1 - fp / n_neg  # FAR at threshold

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
        labels: Binary labels (1 = bonafide, 0 = spoof).
        c_miss: Cost of miss (false rejection).
        c_fa: Cost of false alarm (false acceptance).
        p_target: Prior probability of target (bonafide).

    Returns:
        Minimum DCF value.
    """
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 1.0

    # Sort scores
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # Compute miss and FA rates at each threshold
    fn = np.cumsum(sorted_labels == 1)
    p_miss = fn / n_pos

    fp = n_neg - np.cumsum(sorted_labels == 0)
    p_fa = fp / n_neg

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
        scores: Log-likelihood ratio scores.
        labels: Binary labels (1 = bonafide, 0 = spoof).

    Returns:
        Cllr value.
    """
    # Separate scores by class
    bonafide_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]

    if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
        return 1.0

    # Compute Cllr components
    # For bonafide: average of log2(1 + e^(-score))
    # For spoof: average of log2(1 + e^(score))

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
        scores: Detection scores.
        labels: Binary labels.
        threshold: Decision threshold.
        c_miss: Cost of miss.
        c_fa: Cost of false alarm.
        p_target: Prior probability of target.

    Returns:
        Actual DCF value.
    """
    predictions = (scores >= threshold).astype(int)

    # Compute error rates
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 1.0

    p_miss = np.sum((predictions == 0) & (labels == 1)) / n_pos
    p_fa = np.sum((predictions == 1) & (labels == 0)) / n_neg

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
