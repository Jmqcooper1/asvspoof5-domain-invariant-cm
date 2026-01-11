# ASVspoof 5 Domain-Invariant CM (Track 1)

Domain-invariant speech deepfake detection on ASVspoof 5 Track 1 using Domain-Adversarial Neural Networks (DANN) with mechanistic interpretability analysis.

## Overview

This repository implements:

- **ERM vs DANN comparison** for domain-invariant deepfake detection
- **Two SSL backbones**: WavLM Base+ and Wav2Vec 2.0 Base
- **Multi-head domain discriminator** for CODEC and CODEC_Q domains
- **Mechanistic analyses**: layer-wise domain probes, CKA, limited activation patching
- **Non-semantic baseline**: TRILLsson embeddings + classifier (LFCC+GMM fallback)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Quickstart: Full Pipeline](#quickstart-full-pipeline)
5. [Detailed Usage](#detailed-usage)
6. [Repository Structure](#repository-structure)
7. [Configuration](#configuration)
8. [Reproducibility](#reproducibility)
9. [Testing](#testing)
10. [Citation](#citation)

---

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- ~100GB disk space for full ASVspoof 5 dataset
- GPU with 16GB+ VRAM recommended (A100/H100 for fastest training)

### Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

---

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/asvspoof5-domain-invariant-cm.git
cd asvspoof5-domain-invariant-cm
```

### Step 2: Create virtual environment and install dependencies

```bash
# Create venv and install all dependencies (including dev)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Verify installation
python -c "import asvspoof5_domain_invariant_cm; print('✓ Installation successful')"
```

### Step 3: Run tests to verify everything works

```bash
pytest tests/ -v
```

You should see all tests pass (60+ tests).

---

## Dataset Setup

### Option A: Download from Zenodo (Recommended)

The ASVspoof 5 dataset is hosted on Zenodo. You need to:

1. Create a Zenodo account and accept the data license
2. Download the data files

```bash
# Set your data directory
export ASVSPOOF5_ROOT=/path/to/your/data/asvspoof5
mkdir -p $ASVSPOOF5_ROOT

# Download protocols and minimal subset (for development)
bash scripts/download_asvspoof5.sh

# OR download full dataset (~100GB)
bash scripts/download_asvspoof5.sh --full
```

### Option B: Manual Download

1. Go to [Zenodo ASVspoof 5](https://zenodo.org/records/14498691)
2. Download:
   - `ASVspoof5_protocols.tar.gz` (required)
   - `flac_T_*.tar` (training audio, 5 shards)
   - `flac_D_*.tar` (dev audio, 3 shards)
   - `flac_E_*.tar` (eval audio, 10 shards - optional)
3. Extract to your data directory

### Step 4: Unpack the downloaded files

```bash
# Set the data root
export ASVSPOOF5_ROOT=/path/to/your/data/asvspoof5

# Unpack all archives
bash scripts/unpack_asvspoof5.sh
```

### Expected directory structure

```
$ASVSPOOF5_ROOT/
├── ASVspoof5_protocols/
│   ├── ASVspoof5.train.track_1.tsv
│   ├── ASVspoof5.dev.track_1.tsv
│   └── ASVspoof5.eval.track_1.tsv
├── flac_T/          # Training audio
│   ├── T_0000001.flac
│   └── ...
├── flac_D/          # Dev audio
│   ├── D_0000001.flac
│   └── ...
└── flac_E_eval/     # Eval audio (optional)
    ├── E_0000001.flac
    └── ...
```

### Step 5: Create manifests

```bash
# Ensure ASVSPOOF5_ROOT is set
export ASVSPOOF5_ROOT=/path/to/your/data/asvspoof5

# Create manifests (parquet files with metadata)
python scripts/make_manifest.py

# With validation (checks that audio files exist)
python scripts/make_manifest.py --validate
```

This creates:
- `$ASVSPOOF5_ROOT/manifests/train.parquet`
- `$ASVSPOOF5_ROOT/manifests/dev.parquet`
- `$ASVSPOOF5_ROOT/manifests/eval.parquet` (if eval protocols available)
- `$ASVSPOOF5_ROOT/manifests/codec_vocab.json`
- `$ASVSPOOF5_ROOT/manifests/codec_q_vocab.json`

---

## Quickstart: Full Pipeline

Here's the complete workflow from setup to analysis:

```bash
# 0. Activate environment
source .venv/bin/activate
export ASVSPOOF5_ROOT=/path/to/your/data/asvspoof5

# 1. Create manifests (if not done)
python scripts/make_manifest.py --validate

# 2. Train WavLM ERM baseline
python scripts/train.py --config configs/wavlm_erm.yaml --name wavlm_erm_run1

# 3. Train WavLM DANN
python scripts/train.py --config configs/wavlm_dann.yaml --name wavlm_dann_run1

# 4. Evaluate both models
python scripts/evaluate.py --checkpoint runs/wavlm_erm_run1/checkpoints/best.pt --per-domain
python scripts/evaluate.py --checkpoint runs/wavlm_dann_run1/checkpoints/best.pt --per-domain

# 5. Compare domain probe accuracy (ERM vs DANN)
python scripts/probe_domain.py \
    --erm-checkpoint runs/wavlm_erm_run1/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann_run1/checkpoints/best.pt

# 6. CKA representation similarity analysis
python scripts/run_cka.py \
    --erm-checkpoint runs/wavlm_erm_run1/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann_run1/checkpoints/best.pt

# 7. Activation patching experiment
python scripts/run_patching.py \
    --source runs/wavlm_dann_run1/checkpoints/best.pt \
    --target runs/wavlm_erm_run1/checkpoints/best.pt
```

---

## Detailed Usage

### Training

#### ERM (Baseline)

```bash
# WavLM ERM
python scripts/train.py --config configs/wavlm_erm.yaml --name wavlm_erm

# Wav2Vec 2.0 ERM
python scripts/train.py --config configs/w2v2_erm.yaml --name w2v2_erm
```

#### DANN (Domain Adversarial)

```bash
# WavLM DANN
python scripts/train.py --config configs/wavlm_dann.yaml --name wavlm_dann

# Wav2Vec 2.0 DANN
python scripts/train.py --config configs/w2v2_dann.yaml --name w2v2_dann
```

#### Training options

```bash
# Use AMP (automatic mixed precision) for faster training
python scripts/train.py --config configs/wavlm_dann.yaml --amp

# Resume from checkpoint
python scripts/train.py --config configs/wavlm_dann.yaml --resume runs/wavlm_dann/checkpoints/last.pt

# Override seed
python scripts/train.py --config configs/wavlm_dann.yaml --seed 123

# Use separate config files
python scripts/train.py \
    --train-config configs/train/dann.yaml \
    --model-config configs/model/wavlm_base.yaml \
    --data-config configs/data/asvspoof5_track1.yaml
```

### Evaluation

```bash
# Basic evaluation
python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt

# Per-domain breakdown (CODEC, CODEC_Q)
python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt --per-domain

# With bootstrap confidence intervals
python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt --bootstrap

# Evaluate on eval set (if available)
python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt --split eval

# Generate official score file
python scripts/evaluate.py --checkpoint runs/wavlm_dann/checkpoints/best.pt --scorefile
```

### Interpretability Analysis

#### Domain Probes

```bash
# Single model probe
python scripts/probe_domain.py --checkpoint runs/wavlm_erm/checkpoints/best.pt

# ERM vs DANN comparison (generates comparison plots)
python scripts/probe_domain.py \
    --erm-checkpoint runs/wavlm_erm/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann/checkpoints/best.pt \
    --output-dir analysis/probes

# Probe specific layers
python scripts/probe_domain.py --checkpoint runs/wavlm_erm/checkpoints/best.pt --layers 1,6,11

# With more samples
python scripts/probe_domain.py --checkpoint runs/wavlm_erm/checkpoints/best.pt --n-samples 5000
```

#### CKA Analysis

```bash
python scripts/run_cka.py \
    --erm-checkpoint runs/wavlm_erm/checkpoints/best.pt \
    --dann-checkpoint runs/wavlm_dann/checkpoints/best.pt \
    --output-dir analysis/cka
```

#### Activation Patching

```bash
python scripts/run_patching.py \
    --source runs/wavlm_dann/checkpoints/best.pt \
    --target runs/wavlm_erm/checkpoints/best.pt \
    --layers 9,10,11 \
    --output-dir analysis/patching
```

### Non-Semantic Baselines

#### TRILLsson

```bash
# Install TensorFlow (required for TRILLsson)
uv pip install tensorflow tensorflow-hub

# Extract embeddings
python scripts/extract_trillsson.py --split train --output-dir data/features/trillsson
python scripts/extract_trillsson.py --split dev --output-dir data/features/trillsson

# Train classifier
python scripts/train_trillsson.py --config configs/trillsson_baseline.yaml
```

#### LFCC + GMM (Classical Fallback)

```bash
# Extract LFCC features
python scripts/extract_lfcc.py --split train --output-dir data/features/lfcc
python scripts/extract_lfcc.py --split dev --output-dir data/features/lfcc

# Train GMM
python scripts/train_lfcc_gmm.py --n-components 512 --output-dir runs/lfcc_gmm
```

---

## Repository Structure

```
├── configs/
│   ├── wavlm_erm.yaml          # WavLM + ERM (combined config)
│   ├── wavlm_dann.yaml         # WavLM + DANN (combined config)
│   ├── w2v2_erm.yaml           # Wav2Vec 2.0 + ERM
│   ├── w2v2_dann.yaml          # Wav2Vec 2.0 + DANN
│   ├── trillsson_baseline.yaml # TRILLsson non-semantic baseline
│   ├── data/                   # Data pipeline configs
│   ├── model/                  # Model architecture configs
│   ├── train/                  # Training configs (erm.yaml, dann.yaml)
│   └── eval/                   # Evaluation configs
├── scripts/
│   ├── download_asvspoof5.sh   # Download dataset from Zenodo
│   ├── unpack_asvspoof5.sh     # Unpack downloaded archives
│   ├── make_manifest.py        # Create parquet manifests
│   ├── train.py                # Training entrypoint
│   ├── evaluate.py             # Evaluation entrypoint
│   ├── probe_domain.py         # Layer-wise domain probes
│   ├── run_cka.py              # CKA analysis
│   ├── run_patching.py         # Activation patching
│   ├── extract_trillsson.py    # TRILLsson embedding extraction
│   ├── train_trillsson.py      # TRILLsson classifier training
│   ├── extract_lfcc.py         # LFCC feature extraction
│   └── train_lfcc_gmm.py       # LFCC-GMM baseline
├── src/asvspoof5_domain_invariant_cm/
│   ├── data/                   # Dataset, audio loading, collation
│   ├── models/                 # Backbones, heads, DANN, ERM models
│   ├── training/               # Training loop, schedulers, losses
│   ├── evaluation/             # Metrics (EER, minDCF, Cllr) and reports
│   ├── analysis/               # Probes, CKA, activation patching
│   └── utils/                  # Config, paths, I/O utilities
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── runs/                       # Experiment outputs (created during training)
```

---

## Configuration

All experiments are driven by YAML configs. You can use combined configs or modular ones.

### Combined configs (recommended)

```yaml
# configs/wavlm_dann.yaml
model:
  backbone:
    name: microsoft/wavlm-base-plus
    freeze: true
  layer_selection:
    method: last_k
    k: 4
  pooling:
    method: stats
  projection:
    hidden_dim: 512
    out_features: 256
  classifier:
    num_classes: 2
  discriminator:
    hidden_dim: 256

training:
  method: dann
  optimizer:
    type: adamw
    lr: 1e-4
  epochs: 10
  batch_size: 32
  dann:
    lambda_: 1.0
    lambda_schedule: linear

data:
  sample_rate: 16000
  max_duration: 4.0
```

### Modular configs

```bash
python scripts/train.py \
    --train-config configs/train/dann.yaml \
    --model-config configs/model/wavlm_base.yaml \
    --data-config configs/data/asvspoof5_track1.yaml
```

---

## Reproducibility

Each training run saves to `runs/{exp_name}/`:

```
runs/wavlm_dann_run1/
├── config_resolved.yaml    # Full resolved config
├── checkpoints/
│   ├── best.pt             # Best validation checkpoint
│   └── last.pt             # Final checkpoint
├── train_log.jsonl         # Training logs (loss, metrics per step)
├── metrics_dev.json        # Final dev metrics
├── codec_vocab.json        # CODEC label vocabulary
└── codec_q_vocab.json      # CODEC_Q label vocabulary
```

### Seeds

Set seeds for reproducibility:

```bash
python scripts/train.py --config configs/wavlm_dann.yaml --seed 42
```

Or in config:

```yaml
seed: 42
```

---

## Testing

Run all tests:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=asvspoof5_domain_invariant_cm --cov-report=html
```

Test categories:
- `test_protocol_parse.py` - Protocol parsing and manifest creation
- `test_dataset_shapes.py` - Dataset and batch shapes
- `test_models.py` - Model components (GRL, pooling, heads, DANN/ERM)
- `test_losses.py` - Loss functions
- `test_metrics.py` - Evaluation metrics (EER, minDCF, Cllr)
- `test_config.py` - Config loading and merging
- `test_training.py` - Training components and schedulers
- `test_analysis.py` - Analysis tools (CKA, probes)

---

## Key Design Decisions

### Label Convention
- **bonafide = 0, spoof = 1**
- Score convention: **higher score = more likely bonafide**

### Model Pipeline

```
waveform → backbone → layer_mix → stats_pool → projection → repr (256d)
                                                           ↓
                                                    task_head → logits
                                                           ↓
                                               GRL(repr) → domain_disc → codec/codec_q logits
```

### Domain Handling
- CODEC and CODEC_Q values of `"-"` are normalized to `"NONE"`
- Multi-head discriminator predicts both domains simultaneously
- GRL strength (λ) is schedulable (constant, linear ramp-up, exponential)

### Dataset Notes

**Important**: ASVspoof 5 protocol files are **whitespace-separated** despite the `.tsv` extension. The manifest creation handles this automatically.

---

## Metrics

| Metric | Description | Primary |
|--------|-------------|---------|
| minDCF | Minimum Detection Cost Function (p_target=0.05) | ✓ |
| EER | Equal Error Rate | Secondary |
| Cllr | Log-likelihood ratio cost | Optional |
| actDCF | Actual DCF at operating threshold | Optional |

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{asvspoof5,
  title={ASVspoof 5: Automatic Speaker Verification Spoofing and Deepfake Detection Challenge},
  year={2024}
}

@article{ganin2016dann,
  title={Domain-Adversarial Training of Neural Networks},
  author={Ganin, Yaroslav and others},
  journal={JMLR},
  year={2016}
}
```

---

## License

MIT
