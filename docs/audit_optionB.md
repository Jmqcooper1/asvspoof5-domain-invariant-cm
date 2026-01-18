# Option B Wiring Audit: DANN with Synthetic Codec Augmentation

This document verifies that the DANN training pipeline with synthetic codec augmentation (Option B) is internally consistent and cluster-ready.

## 1. Worker Seeding for Reproducible Augmentation

### Location

`scripts/train.py` lines 76-85:

```python
def _worker_init_fn(worker_id: int) -> None:
    """Seed random for reproducible augmentation in DataLoader workers.
    
    Must be at module level for multiprocessing pickling.
    """
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)
```

### Analysis

- Seeds Python `random` and NumPy per worker using `torch.initial_seed()` plus worker ID
- `torch.initial_seed()` returns the seed set by `torch.manual_seed()` in `set_seed()` (line 330)
- Augmentation in `codec_augment.py` uses `random.random()` and `random.choice()` (lines 380, 387-388)
- This ensures augmentation choices are reproducible across runs with the same seed

### Status: CORRECT

The worker seeding correctly covers Python `random` which is used by the augmentor.

---

## 2. Domain Head Sizes in DANN Mode

### Location

`scripts/train.py` lines 463-474:

```python
# Build model - use synthetic vocab sizes for DANN with augmentation
if augmentor is not None:
    num_codecs = len(SYNTHETIC_CODEC_VOCAB)      # 6 classes
    num_codec_qs = len(SYNTHETIC_QUALITY_VOCAB)  # 6 classes
    logger.info(
        f"DANN with augmentation: domain discriminator sizes "
        f"num_codecs={num_codecs}, num_codec_qs={num_codec_qs}"
    )
else:
    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)
```

### Synthetic Vocabularies

From `src/asvspoof5_domain_invariant_cm/data/codec_augment.py`:

```python
SYNTHETIC_CODEC_VOCAB = {
    "NONE": 0,
    "MP3": 1,
    "AAC": 2,
    "OPUS": 3,
    "SPEEX": 4,
    "AMR": 5,
}  # 6 classes

SYNTHETIC_QUALITY_VOCAB = {
    "NONE": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}  # 6 classes
```

### Analysis

- When augmentation is enabled, DANN uses synthetic vocab sizes (6, 6)
- This is correct because training labels come from augmentation, not manifest
- **Issue identified**: Saved vocab files are still manifest vocabs (lines 365-366), not synthetic vocabs

### Status: PARTIAL - Requires fix for saved vocabs

---

## 3. Domain Labels Used for Loss

### Location

`src/asvspoof5_domain_invariant_cm/training/loop.py` lines 109-115:

```python
# Use augmented domain labels when available (for DANN with synthetic augmentation)
if "y_codec_aug" in batch and batch["y_codec_aug"] is not None:
    y_codec = batch["y_codec_aug"].to(device)
    y_codec_q = batch["y_codec_q_aug"].to(device)
else:
    y_codec = batch["y_codec"].to(device)
    y_codec_q = batch["y_codec_q"].to(device)
```

### Data Flow

1. `ASVspoof5Dataset.__getitem__()` applies augmentation and sets `y_codec_aug`, `y_codec_q_aug` (lines 203-209)
2. `AudioCollator` preserves these in the batch (lines 211-213)
3. Training loop uses augmented labels when present

### CODEC_Q Masking

`src/asvspoof5_domain_invariant_cm/training/losses.py` lines 146-158:

```python
# Mask CODEC_Q loss when codec is NONE (quality undefined for uncoded)
if self.mask_codec_q_for_none:
    mask = codec_labels != self.none_codec_id
    if mask.any():
        l_codec_q = self.codec_q_loss(
            codec_q_logits[mask],
            codec_q_labels[mask],
        )
    else:
        # All samples are uncoded, no CODEC_Q loss
        l_codec_q = torch.tensor(0.0, device=task_logits.device)
```

- `none_codec_id=0` matches `SYNTHETIC_CODEC_VOCAB["NONE"]`
- Quality loss is masked when codec is NONE (correct)

### Status: CORRECT

---

## 4. Validation Domain Label Issue

### Problem

Validation in `loop.py` lines 332-333 uses manifest labels:

```python
y_codec = batch["y_codec"].to(device)
y_codec_q = batch["y_codec_q"].to(device)
```

But dev set has only one domain value (`NONE` encoded as the manifest vocab ID, which is 0 in manifest but the head expects synthetic vocab).

### Impact

- Dev set has `CODEC=NONE` for 100% of samples (see `docs/asvspoof5_domains.md`)
- Domain discriminator accuracy on dev is trivial (always predicts NONE)
- This doesn't break training but provides no useful validation signal for domain heads

### Mitigation

Domain losses/accuracies during validation are logged but not used for early stopping (which uses EER/minDCF). The domain metrics on dev are informational only.

### Status: ACCEPTABLE - No action needed

---

## 5. Saved Vocab Files

### Current Behavior

`scripts/train.py` lines 364-366:

```python
# Copy vocabs to run dir
shutil.copy(manifests_dir / "codec_vocab.json", run_dir / "codec_vocab.json")
shutil.copy(manifests_dir / "codec_q_vocab.json", run_dir / "codec_q_vocab.json")
```

### Problem

When augmentation is enabled, the saved vocabs are manifest vocabs (size varies by splits), but the model was built with synthetic vocabs (size 6, 6). This causes a mismatch when loading the model in `evaluate.py`.

### Fix Required

When `augmentor is not None`, save synthetic vocabs instead:

```python
if augmentor is not None:
    # Save synthetic vocabs for DANN with augmentation
    with open(run_dir / "codec_vocab.json", "w") as f:
        json.dump(SYNTHETIC_CODEC_VOCAB, f, indent=2)
    with open(run_dir / "codec_q_vocab.json", "w") as f:
        json.dump(SYNTHETIC_QUALITY_VOCAB, f, indent=2)
else:
    shutil.copy(manifests_dir / "codec_vocab.json", run_dir / "codec_vocab.json")
    shutil.copy(manifests_dir / "codec_q_vocab.json", run_dir / "codec_q_vocab.json")
```

### Status: REQUIRES FIX

---

## 6. Fail-Fast Checks for Domain Diversity

### Location

`src/asvspoof5_domain_invariant_cm/training/loop.py` lines 126-135:

```python
# Fail-fast: DANN requires domain diversity in early training
if batch_idx < 10:
    unique_codecs = y_codec.unique().numel()
    if unique_codecs < 2:
        raise RuntimeError(
            f"DANN requires domain diversity but batch {batch_idx} has only "
            f"{unique_codecs} unique codec(s). Check: augmentor wired? "
            f"ffmpeg available? supported_codecs>=2? codec_prob>0?"
        )
```

And lines 222-227:

```python
# Fail if augmentation rate is near zero after sufficient steps
if batch_idx >= 500 and aug_rate < 0.05:
    raise RuntimeError(
        f"Augmentation rate {aug_rate:.1%} < 5% after {batch_idx} steps. "
        f"DANN requires domain diversity. Check codec_prob and ffmpeg codec support."
    )
```

### Status: CORRECT

Training will fail fast if augmentation isn't working.

---

## 7. Codec Support Check

### Location

`scripts/train.py` lines 390-396:

```python
if augmentor is not None:
    # Critical: DANN requires domain diversity
    if len(augmentor.supported_codecs) < 2:
        raise RuntimeError(
            f"DANN requires >=2 supported codecs, got {len(augmentor.supported_codecs)}. "
            ...
        )
```

### Status: CORRECT

---

## Summary

| Component | Status | Action |
|-----------|--------|--------|
| Worker seeding | CORRECT | None |
| Domain head sizes | CORRECT | None |
| Training domain labels | CORRECT | None |
| CODEC_Q masking | CORRECT | None |
| Validation domain labels | ACCEPTABLE | None (informational only) |
| Saved vocab files | FIXED | Synthetic vocabs now saved when augmentor enabled |
| Fail-fast checks | CORRECT | None |
| Codec support check | CORRECT | None |

## Applied Fixes

1. **Synthetic vocab saving** (`scripts/train.py`): When augmentation is enabled, synthetic vocabs are saved to the run directory instead of manifest vocabs.

2. **Unbound variable safety** (`src/.../data/codec_augment.py`): Temp paths initialized to `None` before try blocks to prevent `UnboundLocalError` in finally.

3. **Corrupted cache handling** (`src/.../data/codec_augment.py`): Corrupted cache files are now deleted and re-augmented.

4. **Torch seeding in workers** (`scripts/train.py`): Added `torch.manual_seed()` to `_worker_init_fn` for full reproducibility.

---

## Readiness Checklist

### Pre-flight (before submitting jobs)

- [ ] `.env` exists with `ASVSPOOF5_ROOT` pointing to dataset location
- [ ] All 19 tarballs present under `$ASVSPOOF5_ROOT`:
  - `ASVspoof5_protocols.tar.gz`
  - `flac_T_aa.tar` through `flac_T_ae.tar` (5 files)
  - `flac_D_aa.tar` through `flac_D_ac.tar` (3 files)
  - `flac_E_aa.tar` through `flac_E_aj.tar` (10 files)
- [ ] `module spider FFmpeg` shows available version
- [ ] `./scripts/jobs/submit_all.sh --dry-run` succeeds without errors

### During training (automatic checks)

- First 10 batches: fail-fast if `unique_codecs < 2`
- After 500 steps: fail if augmentation rate < 5%
- Codec support check: fail if `supported_codecs < 2`

### Ready for Full Run

**YES** - All identified issues have been fixed. The pipeline is ready for cluster submission.

See `docs/cluster_quickstart.md` for exact Snellius commands.
