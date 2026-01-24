# Reproducibility Guide

This document describes how to ensure reproducible experiments.

## Random Seed Management

### Seed Configuration

Set the seed in your config file or via CLI:

```yaml
# In config file
seed: 42
deterministic: true
```

```bash
# Via CLI (overrides config)
python scripts/train.py --config configs/wavlm_dann.yaml --seed 42
```

### Seeding Policy

The `set_seed()` function seeds all random number generators:

```python
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### DataLoader Worker Seeding

Codec augmentation uses `random.random()` which is not automatically seeded in DataLoader workers.

We use a `worker_init_fn` to ensure reproducible augmentation:

```python
def worker_init_fn(worker_id: int) -> None:
    """Seed random for reproducible augmentation in workers."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
```

This is automatically applied in `scripts/train.py`.

---

## Deterministic Evaluation

### Training vs Evaluation Mode

| Mode | Temporal Processing | Purpose |
|------|---------------------|---------|
| Train | Random crop | Data augmentation |
| Eval | Center crop | Deterministic inference |

The `crop_or_pad()` function handles this:

```python
def crop_or_pad(waveform, n_samples, mode="eval"):
    if current_len > n_samples:
        if mode == "train":
            start = torch.randint(0, current_len - n_samples + 1, (1,)).item()
        else:  # eval
            start = (current_len - n_samples) // 2  # Center crop
        waveform = waveform[..., start : start + n_samples]
```

### Evaluation Consistency

For the same model checkpoint:
- Same seed + eval mode → identical scores
- Different inference runs → identical results

---

## Augmentation Caching

### Cache Determinism

When caching is enabled, augmented files are stored for reuse:

```yaml
augmentation:
  enabled: true
  cache_dir: /path/to/cache
```

**Cache key formula:**
```python
cache_key = MD5(audio_path)[:12] + "_" + codec + "_" + quality
# Example: "a1b2c3d4e5f6_MP3_3.flac"
```

**Determinism guarantee:**
- Same input file + same codec + same quality → same cached output
- Cache is populated on first epoch, reused thereafter
- Atomic writes prevent corruption (`os.replace()`)

### Cache Invalidation

The cache does **not** automatically invalidate if:
- ffmpeg version changes
- Codec parameters change (beyond quality level)

To force re-augmentation, delete the cache directory.

---

## Multi-Seed Experiments

For statistical reliability, run experiments with multiple seeds:

```bash
#!/bin/bash
# run_multi_seed.sh

for seed in 42 123 456 789 1234; do
    python scripts/train.py \
        --config configs/wavlm_dann.yaml \
        --seed $seed \
        --name wavlm_dann_seed${seed}
done
```

### Aggregating Results

After training, aggregate metrics across seeds:

```python
import json
from pathlib import Path

seeds = [42, 123, 456, 789, 1234]
metrics = []

for seed in seeds:
    run_dir = Path(f"runs/wavlm_dann_seed{seed}")
    with open(run_dir / "metrics_train.json") as f:
        m = json.load(f)
        metrics.append(m["best_eer"])

print(f"EER: {np.mean(metrics):.4f} +/- {np.std(metrics):.4f}")
```

---

## Checkpoint Reproducibility

### Saved State

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Epoch number
- Metrics at save time
- Config used for training

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "metrics": metrics,
    "config": config,
}
```

### Resuming Training

To resume from checkpoint:

```bash
python scripts/train.py \
    --config configs/wavlm_dann.yaml \
    --resume runs/wavlm_dann/checkpoints/last.pt
```

**Note:** Resuming resets the random state from the checkpoint, not the original seed.

---

## Environment Reproducibility

### Record Environment

Each run logs:
- Git commit hash
- Python version
- PyTorch version
- GPU type and count

```python
context = {
    "git": {"commit": get_git_commit()},
    "hardware": {"gpu_name": torch.cuda.get_device_name(0)},
    "python": sys.version,
    "pytorch": torch.__version__,
}
```

### Lock Dependencies

Use `uv.lock` to pin exact dependency versions:

```bash
# Install exact versions from lock file
uv sync

# Update lock file after adding dependencies
uv lock
```

---

## Known Non-Determinism Sources

Some operations are not fully deterministic even with seeds set:

| Source | Impact | Mitigation |
|--------|--------|------------|
| CuDNN auto-tuner | Minor variance | `cudnn.deterministic = True` |
| Multi-GPU data parallel | Gradient order | Use single GPU or DDP |
| Floating-point atomics | Numerical noise | Accept small variance |
| Augmentation timing | Order of samples | Fix worker count |

### Acceptable Variance

For a well-seeded run:
- EER variance across identical runs: < 0.1%
- Loss variance: < 0.01%

If variance exceeds this, check:
1. `deterministic: true` in config
2. Same number of workers
3. No concurrent GPU processes

---

## Experiment Tracking

### Local Logs

Each run saves to `runs/{name}/`:
```
runs/wavlm_dann_seed42/
├── config_resolved.yaml    # Full config
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── train_log.jsonl         # Per-epoch metrics
└── metrics_train.json      # Final metrics
```

### Wandb Integration

For cloud tracking:

```bash
python scripts/train.py \
    --config configs/wavlm_dann.yaml \
    --wandb \
    --wandb-project asvspoof5-dann
```

Wandb automatically logs:
- All config parameters
- Training curves
- System metrics (GPU memory, CPU)
- Artifacts (optional)

#### Wandb step semantics (important)

This repo logs **two different cadences**:

- **Step-level training metrics**: `train/step_*` with x-axis `train/global_step`
- **Epoch-level metrics**: `train/*` and `val/*` with x-axis `epoch`

Implementation detail:

- `train/step_*` logs include a `train/global_step` field and are bound via `wandb.define_metric("train/step_*", step_metric="train/global_step")`.
- `train/*` and `val/*` logs include an `epoch` field and are bound via `wandb.define_metric(..., step_metric="epoch")`.
- Epoch logs do **not** pass an explicit `step=` argument to `wandb.log()`, avoiding W&B’s monotonic step constraint when step-level logs have already advanced the internal step counter.

If you see warnings like \"Tried to log to step 0 that is less than the current step ...\", it means epoch metrics were being logged with an explicit `step` and W&B is dropping them.

#### Training stability knobs (early-stop + plateau)

These keys live under `training:` in `configs/train/{erm,dann}.yaml`:

- `min_delta`: minimum improvement in the monitored metric to reset early-stopping patience.
- `train_loss_threshold`: if train loss stays below this threshold, the model is considered to be memorizing training data.
- `plateau_patience`: number of consecutive epochs below `train_loss_threshold` before stopping.
