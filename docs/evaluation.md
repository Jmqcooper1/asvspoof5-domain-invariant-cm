# Evaluation Protocol

This document describes the evaluation metrics and protocols used for ASVspoof 5 Track 1.

## Primary Metrics

### minDCF (Minimum Detection Cost Function)

The primary metric for ASVspoof 5 Track 1. It measures the minimum cost of a detection system given:

- Cost of false alarm (Cfa)
- Cost of miss (Cmiss)
- Prior probability of spoof (Pspoof)

```python
def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    p_target: float = 0.05,
) -> float:
    """Compute minimum Detection Cost Function."""
    # Sort scores and compute FRR/FAR at each threshold
    # Find threshold that minimizes DCF
    ...
```

### EER (Equal Error Rate)

Secondary metric. The point where False Acceptance Rate equals False Rejection Rate.

```python
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate and threshold.
    
    Args:
        scores: Detection scores (higher = more likely bonafide).
        labels: Binary labels (0 = bonafide, 1 = spoof).
    """
    ...
```

## Calibration-Aware Metrics (Optional)

### Cllr (Log-Likelihood Ratio Cost)

Measures how well-calibrated the scores are as likelihood ratios.

### actDCF (Actual DCF)

DCF computed at a fixed threshold (rather than the optimal one).

## Per-Domain Evaluation

To assess domain robustness, we report metrics broken down by:

1. **CODEC:** Performance per codec type
2. **CODEC_Q:** Performance per codec quality setting
3. **Combined:** CODEC x CODEC_Q combinations

### Important: Per-Domain Breakdown Scope

**Per-domain evaluation is meaningful only on the eval split.**

| Split | CODEC diversity | Per-domain breakdown |
|-------|----------------|---------------------|
| Train | None (all `"-"`) | Trivial (single domain) |
| Dev | None (all `"-"`) | Trivial (single domain) |
| Eval | C01-C11 + uncoded | Meaningful |

For train/dev validation during training, only overall metrics (EER, minDCF) are reported.
Per-domain breakdown tables are generated for eval set analysis only.

### Domain Normalization for Reporting

When generating per-domain tables, domain values are normalized:
- `"-"` (train/dev uncoded) → `"NONE"`
- `"0"` (eval uncoded CODEC_Q) → `"NONE"`

This ensures consistent domain labels across plots and tables.

```python
def evaluate_per_domain(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "key",
    domain_col: str = "codec",
) -> pd.DataFrame:
    """Compute metrics for each domain value."""
    results = []
    for domain_val, group in df.groupby(domain_col):
        eer, _ = compute_eer(group[score_col], group[label_col] == "bonafide")
        min_dcf = compute_min_dcf(group[score_col], group[label_col] == "bonafide")
        results.append({
            "domain": domain_val,
            "n_samples": len(group),
            "eer": eer,
            "min_dcf": min_dcf,
        })
    return pd.DataFrame(results)
```

## Domain Invariance Metrics

### Domain Probe Accuracy

Train a linear classifier to predict domain labels from frozen embeddings:

- Lower accuracy = more domain-invariant representations
- Compute per layer to localize domain information

```python
def train_domain_probe(
    embeddings: np.ndarray,
    domain_labels: np.ndarray,
) -> float:
    """Train linear probe, return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, embeddings, domain_labels, cv=5)
    return scores.mean()
```

### In-Domain vs Out-of-Domain Gap

```
gap = EER_out_of_domain - EER_in_domain
```

Smaller gap = better generalization.

## Statistical Reliability

### Bootstrap Confidence Intervals

```python
def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Compute metric with bootstrap confidence interval."""
    ...
```

### Multiple Seeds

Run experiments with different random seeds and report:
- Mean metric
- Standard deviation
- Min/Max across seeds

## Official Evaluation Package

Use the official ASVspoof evaluation tools for final results:
- https://github.com/asvspoof-challenge

This ensures comparability with other published results.
