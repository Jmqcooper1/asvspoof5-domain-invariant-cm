# Snellius Cluster Quickstart

This guide provides exact commands to run the full pipeline on the Snellius (SURF) HPC cluster.

## Prerequisites

- Access to Snellius with project space at `/projects/prjs1904`
- Wandb API key (optional, for experiment tracking)

## Step 1: Clone Repository

```bash
cd /projects/prjs1904
git clone https://github.com/your-username/asvspoof5-domain-invariant-cm.git
cd asvspoof5-domain-invariant-cm
```

## Step 2: Create Environment File

Create a `.env` file in the repo root:

```bash
cat > .env << 'EOF'
# Dataset location (persistent project storage)
ASVSPOOF5_ROOT=/projects/prjs1904/asvspoof5-data

# HuggingFace cache (scratch storage, auto-deleted after 14 days if unused)
HF_HOME=/scratch-shared/$USER/.cache/huggingface

# Wandb (optional - auto-enabled if API key is set)
# Leave empty or remove to disable wandb logging
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=asvspoof5-dann

# FFmpeg module version (change if needed)
FFMPEG_MODULE=FFmpeg/7.1.1-GCCcore-14.2.0
EOF
```

## Step 3: Download Dataset

The full dataset is required for thesis runs. Download on an interactive node:

```bash
# Request interactive session on staging partition
srun --partition=staging --time=12:00:00 --cpus-per-task=4 --pty bash

# Load .env
set -a && source .env && set +a

# Create data directory
mkdir -p "$ASVSPOOF5_ROOT"

# Download full dataset (~100GB)
bash scripts/download_asvspoof5.sh --full --ipv4

# Unpack archives
bash scripts/unpack_asvspoof5.sh

# Exit interactive session
exit
```

### Dataset Requirements

| Tarball | Count | Size | Required for |
|---------|-------|------|--------------|
| `ASVspoof5_protocols.tar.gz` | 1 | ~1MB | All |
| `flac_T_*.tar` | 5 (aa-ae) | ~30GB | Training |
| `flac_D_*.tar` | 3 (aa-ac) | ~20GB | Validation |
| `flac_E_*.tar` | 10 (aa-aj) | ~50GB | Per-domain eval |

**Total: ~100GB**

For a thesis run with per-CODEC evaluation tables, **all tarballs are required**.

### Expected Directory Structure

After unpacking:

```
$ASVSPOOF5_ROOT/
├── ASVspoof5_protocols/
│   ├── ASVspoof5.train.tsv
│   ├── ASVspoof5.dev.track_1.tsv
│   └── ASVspoof5.eval.track_1.tsv
├── flac_T/          # ~182k training files
├── flac_D/          # ~141k dev files
└── flac_E_eval/     # ~681k eval files
```

## Step 4: Submit Pipeline

From the repo root, submit all jobs with dependency management:

```bash
# Dry-run to preview jobs
./scripts/jobs/submit_all.sh --dry-run

# Submit all jobs
./scripts/jobs/submit_all.sh
```

This submits:

1. **StageDataset** - Unpack tarballs, create manifests, validate paths (12h)
2. **Setup** - Download SSL models to HF cache (2h)
3. **Training** - 4 parallel jobs: WavLM ERM/DANN, W2V2 ERM/DANN (24h each)
4. **Baselines** - TRILLsson + LFCC-GMM (8h, parallel with training)
5. **Evaluation** - Evaluate all models on dev (4h, after training)
6. **Analysis** - Domain probes, CKA, patching (8h, after training)
7. **HeldOut** - Held-out codec experiment (48h, after setup)

### Job Dependency Chain

```
StageDataset → Setup → Training (4 parallel) → Evaluation → Analysis
                  ↘ Baselines (parallel)
                  ↘ HeldOut (independent)
```

## Step 5: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View job output
tail -f scripts/jobs/out/WavLM_DANN_*.out

# Cancel all jobs
scancel $(squeue -u $USER -h -o %i)
```

## Step 6: Collect Results

After completion, results are in:

```
runs/
├── wavlm_erm/
│   ├── checkpoints/best.pt
│   ├── eval_dev/metrics.json
│   └── train_log.jsonl
├── wavlm_dann/
├── w2v2_erm/
├── w2v2_dann/
├── analysis/
│   ├── probes_wavlm/
│   ├── cka_wavlm/
│   └── patching_wavlm/
└── held_out_codec/
```

## Troubleshooting

### FFmpeg Not Found

DANN training requires FFmpeg for codec augmentation. If jobs fail with "ffmpeg not found":

```bash
# Check available FFmpeg modules
module spider FFmpeg

# Update FFMPEG_MODULE in .env to match available version
echo "FFMPEG_MODULE=FFmpeg/6.1-GCCcore-12.3.0" >> .env
```

### Dataset Staging Failed

If `stage_dataset.job` fails:

1. Check tarballs exist: `ls $ASVSPOOF5_ROOT/*.tar*`
2. Check disk quota: `myquota`
3. Re-run staging manually: `sbatch scripts/jobs/stage_dataset.job`

### Model Download Timeout

If SSL model downloads timeout, increase HF_HOME cache timeout or pre-download:

```bash
srun --partition=gpu --time=1:00:00 --pty bash
source .env
python -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-base-plus')"
exit
```

## Environment Variables Reference

| Variable | Purpose | Default |
|----------|---------|---------|
| `ASVSPOOF5_ROOT` | Dataset location | **Required** |
| `HF_HOME` | HuggingFace cache | `/scratch-shared/$USER/.cache/huggingface` |
| `WANDB_API_KEY` | Wandb authentication | None (wandb disabled if unset) |
| `WANDB_PROJECT` | Wandb project name | `asvspoof5-dann` |
| `FFMPEG_MODULE` | FFmpeg module to load | `FFmpeg/7.1.1-GCCcore-14.2.0` |

**Note**: Wandb is auto-enabled when `WANDB_API_KEY` is set. No additional flags needed.

## No-Edit Expectation

After following this guide:
- No code changes should be needed
- All paths are configured via `.env`
- Job scripts use relative paths from repo root
