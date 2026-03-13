# Thesis Figures and Tables Specification

> **Auto-generated research document**  
> This spec details all data sources, table/figure requirements, and style guidelines for thesis visualizations.

---

## 1. Data Inventory

### 1.1 Local JSON Files

| File Path | Description | Key Fields |
|-----------|-------------|------------|
| `results/rq3_projection.json` | Projection layer probes (RQ3) | `results.{erm,dann}.codec.{accuracy, accuracy_std, cv_scores}`, `comparison.codec.{reduction, relative_reduction}` |
| `results/domain_probe_results.json` | Domain probe results (WavLM) | `per_layer.{0-11}.{binary,multiclass}.{accuracy, accuracy_std}` |
| `results/domain_probe_wavlm/domain_probe_results.json` | Duplicate of above | Same as above |
| `probe_results/comparison_results.json` | Backbone comparison (WavLM vs W2V2) | `{wavlm,w2v2}.per_layer.{0-11}.{accuracy, accuracy_std}`, `analysis.{wavlm,w2v2}.{mean_accuracy, peak_layer, early/middle/late_layers_mean}` |

### 1.2 Wandb Run IDs

**Primary metrics are stored in Wandb** (project: `mike-cooper-uva/asvspoof5-dann`):

#### Eval Runs (full per-codec metrics)
| Model Key | Run ID | Description |
|-----------|--------|-------------|
| `wavlm_erm` | `7ncrwi99` | WavLM ERM eval on eval set |
| `wavlm_dann` | `aaogtffx` | WavLM DANN eval on eval set |
| `w2v2_erm` | `strsigcn` | Wav2Vec2 ERM eval on eval set |
| `w2v2_dann` | `v4p7t176` | Wav2Vec2 DANN v1 eval on eval set |

#### Dev Runs (for OOD gap calculation)
| Model Key | Run ID | Description |
|-----------|--------|-------------|
| `wavlm_erm` | `txeq8b8p` | WavLM ERM eval on dev set |
| `wavlm_dann` | `5ltu7x9f` | WavLM DANN eval on dev set |
| `w2v2_erm` | `e0mx9gzt` | Wav2Vec2 ERM eval on dev set |
| `w2v2_dann` | `pv9m2g8o` | Wav2Vec2 DANN v1 eval on dev set |

#### Baseline Runs
| Model Key | Run ID | Description |
|-----------|--------|-------------|
| `lfcc_gmm` | `9zrocjqe` | LFCC-GMM baseline |
| `trillsson_logistic` | `5u85m3fu` | TRILLsson + Logistic Regression |
| `trillsson_mlp` | `28qetqki` | TRILLsson + MLP |

#### Probe Comparison Runs
| Model Key | Run ID | Description |
|-----------|--------|-------------|
| `probes_comparison_1` | `evyluap2` | Probe comparison run 1 |
| `probes_comparison_2` | `heo6rsy9` | Probe comparison run 2 |

#### Local Fallback Files (when Wandb unavailable)
```
results/runs/lfcc_gmm_32/eval_eval/metrics.json
results/runs/trillsson_logistic/eval_eval/metrics.json
results/runs/trillsson_mlp/eval_eval/metrics.json
results/runs/w2v2_dann_v2/eval_eval/metrics.json
```

### 1.3 Configuration Files

| Config Path | Purpose | Key Hyperparameters |
|-------------|---------|---------------------|
| `configs/wavlm_erm.yaml` | WavLM ERM training | lr=1e-4, batch=256, epochs=50, patience=10 |
| `configs/wavlm_dann.yaml` | WavLM DANN training | lambda: linear 0.1→0.75, warmup=3 epochs, gradient_clip=0.5 |
| `configs/w2v2_erm.yaml` | Wav2Vec2 ERM training | lr=1e-4, batch=256 |
| `configs/w2v2_dann.yaml` | Wav2Vec2 DANN v1 | exponential schedule |
| `configs/w2v2_dann_v2.yaml` | Wav2Vec2 DANN v2 | lambda: linear 0.1→0.85, first_k=6 layer selection |

### 1.4 Wandb Metric Keys

**Overall metrics** (in run summary):
```
eval/{split}/eer
eval/{split}/min_dcf
eval/{split}/auc
eval/{split}/f1_macro
```

**Per-codec metrics** (12 codecs: C01-C11, NONE):
```
eval/eval/codec/{CODEC}/eer
eval/eval/codec/{CODEC}/auc
eval/eval/codec/{CODEC}/f1_macro
```

**Codec names mapping:**
| Code | Codec |
|------|-------|
| C01 | AMR-WB |
| C02 | EVS |
| C03 | G.722 |
| C04 | G.726 |
| C05 | GSM-FR |
| C06 | iLBC |
| C07 | MP3 |
| C08 | Opus |
| C09 | Speex |
| C10 | Vorbis |
| C11 | μ-law |
| NONE | Uncoded |

---

## 2. Visualization Style Guide

Based on analysis of `scripts/plot_rq3_combined.py`:

### 2.1 Matplotlib rcParams

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

### 2.2 Color Palette

```python
COLORS = {
    "wavlm": "#4C72B0",      # Steel blue
    "w2v2": "#DD8452",       # Coral/orange
    "erm": "#E57373",        # Light red/coral
    "dann": "#64B5F6",       # Light blue
    "chance": "#9E9E9E",     # Gray
    
    # Extended palette for models (from notebook)
    'wavlm_erm': '#1f77b4',       # Blue
    'wavlm_dann': '#ff7f0e',      # Orange
    'w2v2_erm': '#2ca02c',        # Green
    'w2v2_dann': '#d62728',       # Red
    'w2v2_dann_v2': '#9467bd',    # Purple
}
```

### 2.3 Model Labels

```python
MODEL_LABELS = {
    'wavlm_erm': 'WavLM ERM',
    'wavlm_dann': 'WavLM DANN',
    'w2v2_erm': 'Wav2Vec2 ERM',
    'w2v2_dann': 'Wav2Vec2 DANN v1',
    'w2v2_dann_v2': 'Wav2Vec2 DANN v2',
}
```

### 2.4 Output Formats

- Save both PNG (300 DPI) and PDF for each figure
- Use `bbox_inches='tight'`, `facecolor='white'`
- Output directory: `figures/` (create if needed)

---

## 3. Tables Specification

### T1: Main Results Table

**Purpose:** Overall comparison of all models with all metrics

**Data Source:** Wandb eval runs + local fallbacks

**Columns:**
| Column | Source Key | Format |
|--------|-----------|--------|
| Model | - | "WavLM" / "Wav2Vec2" / "LFCC-GMM" / "TRILLsson" |
| Method | - | "ERM" / "DANN" / "DANN v2" / "Baseline" / "Logistic" / "MLP" |
| Dev EER | `eval/dev/eer` | %.2f (%) or %.4f (decimal) |
| Eval EER | `eval/eval/eer` | %.2f (%) |
| minDCF | `eval/eval/min_dcf` | %.4f |
| AUC | `eval/eval/auc` | %.4f |
| F1 | `eval/eval/f1_macro` | %.4f |

**Rows (8 models):**
1. WavLM ERM
2. WavLM DANN
3. Wav2Vec2 ERM
4. Wav2Vec2 DANN v1
5. Wav2Vec2 DANN v2
6. LFCC-GMM (baseline)
7. TRILLsson Logistic
8. TRILLsson MLP

**Sample values from notebook:**
- WavLM ERM: Eval EER = 0.0848
- WavLM DANN: Eval EER = 0.0736
- W2V2 ERM: Eval EER = 0.1515
- W2V2 DANN: Eval EER = 0.1437
- LFCC-GMM: Eval EER = 0.4333

---

### T2: Per-Codec EER Comparison

**Purpose:** Show EER per codec for main models with delta columns

**Data Source:** Wandb per-codec metrics

**Columns:**
| Column | Description |
|--------|-------------|
| Codec | Human-readable name (e.g., "MP3", "Uncoded") |
| WavLM ERM | EER for wavlm_erm |
| WavLM DANN | EER for wavlm_dann |
| Δ WavLM | wavlm_erm - wavlm_dann (negative = DANN better) |
| W2V2 ERM | EER for w2v2_erm |
| W2V2 DANN | EER for w2v2_dann |
| Δ W2V2 | w2v2_erm - w2v2_dann |

**Rows:** 12 codecs (C01-C11 + NONE)

**Sample values:**
- C01 (AMR-WB): WavLM ERM = 0.0753
- C07 (MP3): WavLM ERM = 0.1165
- NONE (Uncoded): WavLM ERM = 0.0612

---

### T3: OOD Gap Analysis

**Purpose:** Show generalization gap between dev and eval sets

**Data Source:** Dev runs (for Dev EER) + Eval runs (for Eval EER)

**Columns:**
| Column | Description |
|--------|-------------|
| Model | Model name |
| Method | Training method |
| Dev EER | EER on dev set |
| Eval EER | EER on eval set |
| Gap | Eval EER - Dev EER |
| Gap Reduction | Compared to ERM baseline (%) |

**Calculation:**
```python
gap = eval_eer - dev_eer
gap_reduction = (erm_gap - dann_gap) / erm_gap * 100
```

---

### T4: Projection Probe Results (RQ3)

**Purpose:** Show codec probe accuracy at projection layer (ERM vs DANN)

**Data Source:** `results/rq3_projection.json`

**JSON paths:**
```python
# ERM projection accuracy
data["results"]["erm"]["codec"]["accuracy"]  # 0.434
data["results"]["erm"]["codec"]["accuracy_std"]  # 0.0089

# DANN projection accuracy  
data["results"]["dann"]["codec"]["accuracy"]  # 0.388
data["results"]["dann"]["codec"]["accuracy_std"]  # 0.0090

# Comparison metrics
data["comparison"]["codec"]["reduction"]  # 0.046
data["comparison"]["codec"]["relative_reduction"]  # 0.107 (10.7%)
```

**Columns:**
| Column | Value |
|--------|-------|
| Model | "ERM" / "DANN" |
| Codec Probe Accuracy | mean ± std |
| n_samples | 5000 |
| n_classes | 12 |
| cv_folds | 5 |

**Summary row:**
- Reduction: 0.046 (10.7% relative)
- DANN is more domain-invariant: True

---

### T5: Dataset Statistics

**Purpose:** Document dataset size and codec distribution

**Data Source:** Protocol files / `results/domain_probe_results.json`

**From probe results:**
```python
data["num_original_samples"]  # 5000
data["num_total_samples"]  # 10000 (after augmentation)
data["codecs_available"]  # ["MP3", "AAC", "OPUS"]
data["binary_class_distribution"]  # {0: 5000, 1: 5000}
data["multiclass_class_distribution"]  # {0: 5000, 1: 1694, 2: 1648, 3: 1658}
```

**Columns:**
| Column | Description |
|--------|-------------|
| Split | train / dev / eval |
| Bonafide | Count of genuine samples |
| Spoof | Count of spoofed samples |
| Total | Sum |
| Codecs | List or count of codec types |

**Note:** Full ASVspoof5 stats may need to be computed from protocol files if not in Wandb.

---

### T6: Synthetic Augmentation Coverage

**Purpose:** Document augmentation strategy

**Data Source:** Config files (`configs/wavlm_dann.yaml`)

**From config:**
```yaml
augmentation:
  enabled: true
  codec_prob: 0.5
  codecs: [MP3, AAC, OPUS]
  qualities: [1, 2, 3, 4, 5]
```

**Columns:**
| Column | Value |
|--------|-------|
| Augmentation Probability | 50% |
| Synthetic Codecs | MP3, AAC, OPUS |
| Quality Levels | 1, 2, 3, 4, 5 |
| Purpose | Create domain labels for DANN |

---

### T7: Hyperparameters Table

**Purpose:** Document key hyperparameters for reproducibility

**Data Source:** Config files

**Rows (extract key params):**
| Parameter | WavLM ERM | WavLM DANN | W2V2 ERM | W2V2 DANN v1 | W2V2 DANN v2 |
|-----------|-----------|------------|----------|--------------|--------------|
| Learning Rate | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 5e-5 |
| Batch Size | 256 | 256 | 256 | 256 | 256 |
| Max Epochs | 50 | 50 | 50 | 50 | 50 |
| Patience | 10 | 10 | 10 | 10 | 10 |
| Gradient Clip | 1.0 | 0.5 | 1.0 | 0.5 | 0.5 |
| Lambda Schedule | - | linear 0.1→0.75 | - | exp 0.01→1.0 | linear 0.1→0.85 |
| Warmup Epochs | - | 3 | - | 0 | 3 |
| Layer Selection | weighted k=6 | weighted k=6 | weighted k=6 | weighted k=6 | first_k k=6 |
| Backbone Frozen | Yes | Yes | Yes | Yes | Yes |

---

## 4. Figures Specification

### F1: Per-Codec EER Bar Chart

**Purpose:** Visual comparison of EER across codecs for all models

**Data Source:** Wandb per-codec metrics

**Layout:**
- Grouped bar chart
- X-axis: Codecs (12 groups: C01-C11 + NONE, use human-readable names)
- Y-axis: EER (0 to max, e.g., 0.3)
- Groups: 4 bars per codec (WavLM ERM, WavLM DANN, W2V2 ERM, W2V2 DANN)
- Colors: Use `COLORS` dict for each model

**Style notes:**
- Rotate x-labels 45° if needed
- Add legend outside plot (upper right or below)
- Error bars if available (not in current data)
- Horizontal line at overall average EER for reference

**Figsize:** (14, 6) or (12, 5)

---

### F2: OOD Gap Visualization

**Purpose:** Show Dev vs Eval EER with gap arrows

**Data Source:** Dev and Eval run summaries

**Layout options:**

**Option A: Grouped bar with arrows**
- X-axis: Models (4 groups)
- Y-axis: EER
- Two bars per model: Dev (lighter) and Eval (darker)
- Arrows showing gap between bars
- Annotate gap value

**Option B: Slope chart (slopegraph)**
- Left column: Dev EER
- Right column: Eval EER
- Lines connecting same model
- Color by model
- Steeper slope = worse generalization

**Recommended:** Option A for clarity

**Figsize:** (10, 6)

---

### F3: Score Distributions

**Purpose:** Show separation between bonafide and spoof scores

**Data Source:** Would need raw scores (not in current data)

**Layout:**
- Subplots: 2x2 grid (one per main model)
- Each subplot: KDE or histogram of bonafide vs spoof scores
- Vertical line at threshold (EER operating point)
- Overlap region highlighted

**Note:** This figure requires raw score outputs. Check if `scores.npy` or similar exists in eval output directories. If not available, mark as **NEEDS DATA**.

---

### F4: Training Curves

**Purpose:** Show training dynamics (loss, EER over epochs)

**Data Source:** Wandb run history (not summary)

**Layout:**
- 2 rows: Loss curves, EER curves
- Columns: ERM models, DANN models (or overlay)
- X-axis: Epoch
- Y-axis: Loss / EER

**Wandb API:**
```python
run = api.run(f"{entity}/{project}/{run_id}")
history = run.history()  # DataFrame with per-step metrics
```

**Keys to plot:**
- `train/loss` or `train/task_loss`
- `val/eer` or `dev/eer`
- For DANN: `train/domain_loss`, `train/lambda`

**Note:** May need to query Wandb history API. Mark as **NEEDS WANDB API**.

---

### F5: Lambda Schedule Comparison

**Purpose:** Visualize DANN lambda schedules (v1 exponential vs v2 linear)

**Data Source:** Config files (analytical) or Wandb history (actual)

**Layout:**
- X-axis: Epoch (0 to max_epochs)
- Y-axis: Lambda value (0 to 1)
- Two lines: v1 (exponential), v2 (linear)
- Annotate warmup period, final values

**Analytical computation:**
```python
# From configs:
# v1 (w2v2_dann): exponential 0.01 → 1.0
# v2 (w2v2_dann_v2): linear 0.1 → 0.85, warmup=3
# wavlm_dann: linear 0.1 → 0.75, warmup=3

def linear_schedule(epoch, start, end, warmup, max_epochs):
    if epoch < warmup:
        return start
    progress = (epoch - warmup) / (max_epochs - warmup)
    return start + progress * (end - start)

def exponential_schedule(epoch, start, end, max_epochs, gamma=10):
    progress = epoch / max_epochs
    return end - (end - start) * np.exp(-gamma * progress)
```

**Figsize:** (8, 5)

---

### F6 (Existing): RQ3 Combined Figure

**Purpose:** Two-panel figure for RQ3 (backbone probes + projection probes)

**Already implemented:** `scripts/plot_rq3_combined.py`

**Panels:**
1. LEFT: Backbone layer-wise codec probe accuracy (layers 0-11)
   - Shows WavLM and W2V2 curves
   - Chance level line
2. RIGHT: Projection layer probe (ERM vs DANN bar chart)
   - Can show 2 bars (WavLM only) or 4 bars (both backbones)
   - Reduction annotation

**Data sources:**
- `probe_results/comparison_results.json` (backbone)
- `results/rq3_projection.json` (projection)

---

## 5. Missing Data Assessment

| Item | Status | Action Needed |
|------|--------|---------------|
| Per-codec metrics | ✅ Available | Query from Wandb |
| Overall metrics | ✅ Available | Query from Wandb |
| Dev metrics | ✅ Available | Query from Wandb (dev run IDs) |
| Projection probes | ✅ Available | Load from `results/rq3_projection.json` |
| Backbone probes | ✅ Available | Load from `probe_results/comparison_results.json` |
| Training curves | ⚠️ Needs Wandb API | Query `run.history()` |
| Raw scores | ❌ Not found | Generate by re-running eval with `--save-scores` or similar |
| Dataset statistics | ⚠️ Partial | May need to compute from protocol files |
| Lambda history | ⚠️ Needs Wandb API | Query from run history or compute analytically |

---

## 6. Implementation Checklist

### Tables (generate as LaTeX + Markdown)

- [ ] T1: Main Results — Wandb API query
- [ ] T2: Per-Codec EER — Wandb API query
- [ ] T3: OOD Gap — Combine dev + eval queries
- [ ] T4: Projection Probes — Load JSON
- [ ] T5: Dataset Statistics — Protocol files / hardcode
- [ ] T6: Augmentation Coverage — Extract from configs
- [ ] T7: Hyperparameters — Extract from configs

### Figures (generate as PNG + PDF)

- [ ] F1: Per-Codec EER Bar Chart — Wandb API
- [ ] F2: OOD Gap Visualization — Wandb API
- [ ] F3: Score Distributions — **BLOCKED** (needs raw scores)
- [ ] F4: Training Curves — Wandb history API
- [ ] F5: Lambda Schedule — Analytical from configs
- [ ] F6: RQ3 Combined — **DONE** (existing script)

---

## 7. Script Template

Recommended structure for generation script:

```python
#!/usr/bin/env python3
"""Generate thesis tables and figures.

Usage:
    python scripts/generate_thesis_visuals.py --output-dir outputs/thesis
"""

import argparse
import json
from pathlib import Path
import wandb
import pandas as pd
import matplotlib.pyplot as plt

# Constants from this spec
WANDB_ENTITY = "mike-cooper-uva"
WANDB_PROJECT = "asvspoof5-dann"
EVAL_RUN_IDS = {...}  # From Section 1.2
DEV_RUN_IDS = {...}
COLORS = {...}  # From Section 2.2
STYLE_CONFIG = {...}  # From Section 2.1

def load_wandb_metrics():
    """Load metrics from Wandb API."""
    api = wandb.Api()
    # ... implementation

def generate_t1_main_results(metrics: dict, output_dir: Path):
    """Generate Table T1: Main Results."""
    # ... implementation

def generate_f1_per_codec_bar(metrics: dict, output_dir: Path):
    """Generate Figure F1: Per-Codec EER Bar Chart."""
    plt.rcParams.update(STYLE_CONFIG)
    # ... implementation

# Entry point
if __name__ == "__main__":
    # Parse args, load data, generate outputs
    pass
```

---

## 8. References

- **Existing scripts:**
  - `scripts/plot_rq3_combined.py` — Reference for styling
  - `notebooks/thesis_results.ipynb` — Reference for Wandb queries

- **Data files:**
  - `results/rq3_projection.json`
  - `probe_results/comparison_results.json`
  - `configs/*.yaml`

- **Wandb project:** `mike-cooper-uva/asvspoof5-dann`
