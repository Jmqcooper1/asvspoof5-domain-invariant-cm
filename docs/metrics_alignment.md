# Metrics Alignment

This document specifies the score polarity, metrics reported, and assumptions for the evaluation pipeline.

## Score Convention

**Higher score = more likely bonafide (genuine speech)**

| Component | Score interpretation |
|-----------|---------------------|
| Model output | `softmax(logits)[:, 0]` = P(bonafide) |
| EER threshold | Scores >= threshold → accept as bonafide |
| minDCF threshold | Scores >= threshold → accept as bonafide |

This convention is consistent across:
- `scripts/train.py` (training loop)
- `scripts/evaluate.py` (evaluation)
- `src/asvspoof5_domain_invariant_cm/evaluation/metrics.py`

## Label Convention

| Label | Meaning | Integer value |
|-------|---------|---------------|
| Bonafide | Genuine human speech | 0 |
| Spoof | Synthetic/manipulated | 1 |

Defined in `src/asvspoof5_domain_invariant_cm/data/asvspoof5.py`:

```python
KEY_TO_LABEL = {"bonafide": 0, "spoof": 1}
```

## Primary Metrics

### EER (Equal Error Rate)

The threshold where False Acceptance Rate equals False Rejection Rate.

- Lower is better
- Threshold-independent (evaluates the ranking quality)
- Reported with bootstrap 95% CI when `--bootstrap` flag is used

### minDCF (Minimum Detection Cost Function)

The minimum cost achievable at any operating threshold.

- Lower is better
- Parameters: `c_miss=1.0`, `c_fa=1.0`, `p_target=0.05` (ASVspoof 5 defaults)
- Reported with bootstrap 95% CI when `--bootstrap` flag is used

## Cllr (Log-Likelihood Ratio Cost)

### Status: DISABLED BY DEFAULT

Cllr is implemented in `src/asvspoof5_domain_invariant_cm/evaluation/metrics.py` but **not reported** by `scripts/evaluate.py`.

### Why Cllr is Disabled

Cllr measures calibration quality and assumes scores are **log-likelihood ratios (LLRs)**:

```
LLR = log(P(x | bonafide) / P(x | spoof))
```

Our model outputs `softmax(logits)[:, 0]` which is `P(bonafide | x)`, **not** an LLR. Using Cllr with posterior probabilities instead of LLRs produces meaningless values.

### Enabling Cllr (if needed)

To use Cllr correctly, you must first calibrate scores to LLRs. A common approach:

1. Train a logistic regression on dev set: `LLR = a * score + b`
2. Apply calibration to eval set scores
3. Compute Cllr on calibrated LLRs

This is out of scope for the current codebase. If you need Cllr:

1. Implement calibration in `src/asvspoof5_domain_invariant_cm/evaluation/calibration.py`
2. Add `--calibrate` flag to `scripts/evaluate.py`
3. Report Cllr only on calibrated scores

### Alternative: actDCF

If you need an operating-point-specific metric, use actDCF (actual DCF at a fixed threshold). This is implemented and does not require LLR calibration.

## Metrics Not Reported

| Metric | Status | Reason |
|--------|--------|--------|
| Cllr | Disabled | Requires LLR calibration |
| minCllr | Not implemented | Same as Cllr |
| AUC | Not implemented | Less interpretable for ASVspoof |

## Per-Domain Breakdown

When `--per-domain` flag is used, `scripts/evaluate.py` reports EER and minDCF broken down by:

- CODEC (C01-C11, NONE)
- CODEC_Q (0-8)

**Note**: Per-domain breakdown is only meaningful on the **eval split**. Train/dev have no codec diversity (100% NONE).

## Score File Format

When `--scorefile` flag is used, scores are saved in ASVspoof official format:

```
utterance_id score
T_0000001 0.987654
T_0000002 0.012345
...
```

This format is compatible with the official ASVspoof evaluation tools.
