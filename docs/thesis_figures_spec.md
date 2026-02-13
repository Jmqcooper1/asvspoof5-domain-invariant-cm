# Thesis Figures and Tables Specification

This document specifies all figures and tables to be generated for the thesis on Domain-Invariant Speech Deepfake Detection using DANN.

## Tables

### T1: Main Results Table

**Purpose:** Compare overall performance of all models on dev and eval splits.

**Columns:**
| Model | Backbone | Dev EER (%) | Eval EER (%) | Eval minDCF |
|-------|----------|-------------|--------------|-------------|

**Models:**
- WavLM ERM (baseline)
- WavLM DANN
- W2V2 ERM
- W2V2 DANN

**Data source:** `results/main_results.json` or wandb

---

### T2: Per-Codec EER Comparison

**Purpose:** Show model performance breakdown by codec type on eval set.

**Columns:**
| Codec | WavLM ERM | WavLM DANN | W2V2 ERM | W2V2 DANN |
|-------|-----------|------------|----------|-----------|

**Codecs:** C01-C11 + uncoded (NONE)

**Data source:** `results/per_codec_eer.json`

---

### T3: OOD Gap Analysis

**Purpose:** Quantify the in-domain (dev) vs out-of-domain (eval) generalization gap.

**Columns:**
| Model | Dev EER (%) | Eval EER (%) | Gap | Gap Reduction vs ERM |
|-------|-------------|--------------|-----|---------------------|

**Data source:** Computed from T1

---

### T4: Projection Probe Results

**Purpose:** Show codec probe accuracy on projection layer outputs (RQ3).

**Columns:**
| Backbone | ERM Probe Acc | DANN Probe Acc | Reduction (%) |
|----------|--------------|----------------|---------------|

**Data source:** `results/rq3_projection.json`, `results/rq3_projection_w2v2.json`

---

### T5: Dataset Statistics

**Purpose:** Summarize ASVspoof 5 dataset composition.

**Rows:**
| Split | Bonafide | Spoof | Total | Codecs | Duration (h) |
|-------|----------|-------|-------|--------|--------------|
| Train | ... | ... | ... | - (uncoded) | ... |
| Dev | ... | ... | ... | - (uncoded) | ... |
| Eval | ... | ... | ... | 12 (C01-C11+uncoded) | ... |

**Data source:** `$ASVSPOOF5_ROOT/manifests/*.parquet` or hardcoded from protocol

---

### T6: Synthetic Augmentation Coverage

**Purpose:** Document the synthetic domain augmentation used in DANN training.

**Columns:**
| Codec | Quality Levels | Bitrates |
|-------|---------------|----------|
| MP3 | 4 | 64k, 96k, 128k, 192k |
| AAC | 4 | 64k, 96k, 128k, 192k |
| Opus | 4 | 32k, 64k, 96k, 128k |

**Data source:** `configs/augmentation.yaml` or hardcoded

---

### T7: Hyperparameters

**Purpose:** Document key hyperparameters for reproducibility.

**Sections:**
- Model architecture (backbone, projection dim, classifier heads)
- Training (batch size, learning rate, epochs, optimizer)
- DANN-specific (lambda schedule, domain head architecture)
- Regularization (dropout, weight decay)

**Data source:** `configs/train/wavlm_dann.yaml`, `configs/train/wavlm_erm.yaml`

---

## Figures

### F1: Per-Codec EER Bar Chart

**Purpose:** Visualize per-codec performance for all 4 models.

**Style:**
- Grouped bar chart (4 bars per codec)
- Colors: WavLM ERM, WavLM DANN, W2V2 ERM, W2V2 DANN
- X-axis: Codec names (C01, C02, ..., C11, NONE)
- Y-axis: EER (%)
- Legend: Model names
- Match `plot_rq3_combined.py` style

**Data source:** `results/per_codec_eer.json`

**Output:** `figures/per_codec_eer.{png,pdf}`

---

### F2: OOD Gap Visualization

**Purpose:** Show dev→eval generalization gap and DANN's improvement.

**Style:**
- Paired bar chart or line with arrows
- Show dev EER and eval EER for each model
- Annotate gap reduction percentages
- Highlight DANN vs ERM improvement

**Data source:** Main results (T1)

**Output:** `figures/ood_gap.{png,pdf}`

---

### F3: RQ3 Combined (Existing)

**Purpose:** Domain invariance analysis (backbone probes + projection probes).

**Note:** Already implemented in `scripts/plot_rq3_combined.py`

**Output:** `figures/rq3_combined.{png,pdf}`

---

### F4: Backbone Comparison (Existing)

**Purpose:** Layer-wise codec probe accuracy for WavLM vs W2V2.

**Note:** Already exists as `probe_results/backbone_comparison.png`

---

### F5: Lambda Schedule Ablation

**Purpose:** Compare λ scheduling strategies for DANN.

**Style:**
- Line plot showing λ over training epochs
- v1 (exponential ramp) vs v2 (linear ramp)
- Optionally overlay training loss/EER curves

**Data source:** Training logs or `results/lambda_ablation.json`

**Output:** `figures/lambda_ablation.{png,pdf}`

---

## Output Directory Structure

```
figures/
├── tables/
│   ├── T1_main_results.tex
│   ├── T1_main_results.md
│   ├── T2_per_codec.tex
│   ├── T2_per_codec.md
│   ├── T3_ood_gap.tex
│   ├── T3_ood_gap.md
│   ├── T4_projection_probes.tex
│   ├── T4_projection_probes.md
│   ├── T5_dataset_stats.tex
│   ├── T5_dataset_stats.md
│   ├── T6_augmentation.tex
│   ├── T6_augmentation.md
│   ├── T7_hyperparameters.tex
│   └── T7_hyperparameters.md
├── per_codec_eer.png
├── per_codec_eer.pdf
├── ood_gap.png
├── ood_gap.pdf
├── lambda_ablation.png
├── lambda_ablation.pdf
├── rq3_combined.png  (existing)
└── rq3_combined.pdf  (existing)
```

---

## Color Scheme

Consistent colors across all figures (from `plot_rq3_combined.py`):

```python
COLORS = {
    "wavlm": "#4C72B0",      # Steel blue
    "w2v2": "#DD8452",       # Coral/orange
    "wavlm_erm": "#E57373",  # Light red
    "wavlm_dann": "#64B5F6", # Light blue
    "w2v2_erm": "#FFB74D",   # Light orange
    "w2v2_dann": "#81C784",  # Light green
    "erm": "#E57373",        # Light red
    "dann": "#64B5F6",       # Light blue
    "chance": "#9E9E9E",     # Gray
}
```

---

## Style Guidelines

All figures should follow the style in `scripts/plot_rq3_combined.py`:

```python
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
```
