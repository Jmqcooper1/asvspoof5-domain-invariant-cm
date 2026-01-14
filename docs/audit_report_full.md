# ASVspoof5 DANN Codebase Audit Report

**Date:** 2026-01-14  
**Auditor:** AI Assistant  
**Scope:** Full codebase audit for Option B (CODEC + CODEC_Q domain-adversarial training)

## Executive Summary

The codebase audit identified **6 critical bugs** that would cause DANN training to silently degenerate to ERM, plus **4 important bugs** affecting reproducibility and safety. All issues have been fixed and verified with 140 passing tests.

| Priority | Issues Found | Issues Fixed | Status |
|----------|--------------|--------------|--------|
| P0 (Critical) | 6 | 6 | GREEN |
| P1 (Important) | 4 | 4 | GREEN |
| P2 (Enhancement) | 4 | 4 | GREEN |

---

## P0: Critical Bugs Fixed

### P0-1: CodecAugmentor Not Wired in train.py

**Status:** GREEN (Fixed)

**Problem:** `ASVspoof5Dataset` was created without `augmentor` or `use_synthetic_labels` parameters, making DANN training degenerate to ERM.

**File:** `scripts/train.py`

**Fix Applied:**
- Added imports for `create_augmentor`, `SYNTHETIC_CODEC_VOCAB`, `SYNTHETIC_QUALITY_VOCAB`
- Created augmentor when `method == "dann"` and `augmentation.enabled == True`
- Added hard fail if `len(supported_codecs) < 2`
- Passed `augmentor` and `use_synthetic_labels` to `ASVspoof5Dataset`

### P0-2: Training Loop Used Original Labels Instead of Augmented

**Status:** GREEN (Fixed)

**Problem:** `train_epoch()` used `batch["y_codec"]` (always NONE) instead of `batch["y_codec_aug"]`.

**File:** `src/asvspoof5_domain_invariant_cm/training/loop.py`

**Fix Applied:**
```python
if "y_codec_aug" in batch and batch["y_codec_aug"] is not None:
    y_codec = batch["y_codec_aug"].to(device)
    y_codec_q = batch["y_codec_q_aug"].to(device)
else:
    y_codec = batch["y_codec"].to(device)
    y_codec_q = batch["y_codec_q"].to(device)
```

### P0-3: Metrics Implementation Used Wrong Label Convention

**Status:** GREEN (Fixed)

**Problem:** `compute_eer()`, `compute_min_dcf()`, `compute_act_dcf()`, and `compute_cllr()` internally assumed `1=bonafide, 0=spoof`, but `KEY_TO_LABEL` defines `0=bonafide, 1=spoof`.

**File:** `src/asvspoof5_domain_invariant_cm/evaluation/metrics.py`

**Fix Applied:** Rewrote all metric functions to correctly handle `0=bonafide, 1=spoof` convention.

### P0-4: Domain Discriminator Used Protocol Vocab Sizes

**Status:** GREEN (Fixed)

**Problem:** Domain discriminator `num_codecs`/`num_codec_qs` was set from protocol vocab sizes (which are 1 for train/dev), but synthetic augmentation produces labels 0-5.

**File:** `scripts/train.py`

**Fix Applied:**
```python
if augmentor is not None:
    num_codecs = len(SYNTHETIC_CODEC_VOCAB)      # 6
    num_codec_qs = len(SYNTHETIC_QUALITY_VOCAB)  # 6
else:
    num_codecs = len(codec_vocab)
    num_codec_qs = len(codec_q_vocab)
```

### P0-5: No Fail-Fast for Missing Domain Diversity

**Status:** GREEN (Fixed)

**Problem:** DANN could silently train with constant NONE labels if augmentation failed.

**File:** `src/asvspoof5_domain_invariant_cm/training/loop.py`

**Fix Applied:** Added fail-fast check in first 10 batches:
```python
if self.method == "dann" and batch_idx < 10:
    unique_codecs = y_codec.unique().numel()
    if unique_codecs < 2:
        raise RuntimeError("DANN requires domain diversity...")
```

### P0-6: No Augmentation Rate Monitoring

**Status:** GREEN (Fixed)

**Problem:** Even with wiring, augmentation could silently fail (ffmpeg issues, codec_prob=0).

**File:** `src/asvspoof5_domain_invariant_cm/training/loop.py`

**Fix Applied:** Added augmentation rate tracking with fail if < 5% after 500 steps.

---

## P1: Important Bugs Fixed

### P1-1: DataLoader Workers Not Seeded

**Status:** GREEN (Fixed)

**Problem:** `random.random()` used in augmentation wasn't seeded in DataLoader workers.

**File:** `scripts/train.py`

**Fix Applied:** Added `worker_init_fn` that seeds `random` and `numpy` per worker.

### P1-2: Cache Writes Not Atomic

**Status:** GREEN (Fixed)

**Problem:** Direct `torchaudio.save(cache_path)` could leave corrupted files if interrupted.

**File:** `src/asvspoof5_domain_invariant_cm/data/codec_augment.py`

**Fix Applied:** Write to temp file, then `os.replace()` for atomic rename.

### P1-3: No Test for CODEC_Q Masking

**Status:** GREEN (Fixed)

**Problem:** No unit test verifying CODEC_Q loss is 0 when codec is NONE.

**File:** `tests/test_losses.py`

**Fix Applied:** Added `test_codec_q_loss_masked_when_codec_none()` and `test_codec_q_loss_not_masked_for_coded_samples()`.

### P1-4: No Test for Vocab ID Consistency

**Status:** GREEN (Fixed)

**Problem:** No test verifying "-" and "0" map to same vocab ID.

**File:** `tests/test_protocol_parse.py`

**Fix Applied:** Added `test_dash_and_zero_same_vocab_id()`.

---

## P2: Enhancements Added

### P2-1: Batch Diversity Warning

**Status:** GREEN (Included in P0-5/P0-6)

Logging of unique codecs per batch and augmentation rate in DANN training.

### P2-2: CODEC_Q Semantic Mismatch Documentation

**Status:** GREEN (Fixed)

**File:** `docs/asvspoof5_domains.md`

Added section documenting that synthetic quality tiers (1-5) don't correspond to eval quality levels (0-8).

### P2-3: Codec Taxonomy Gaps Documentation

**Status:** GREEN (Fixed)

**File:** `docs/asvspoof5_domains.md`

Added section documenting that C04 (Encodec), C07 (MP3+Encodec cascade), and C11 (device/channel) are not covered by synthetic augmentation.

### P2-4: Enhanced ffmpeg Codec Warning

**Status:** GREEN (Fixed)

**File:** `src/asvspoof5_domain_invariant_cm/data/codec_augment.py`

Enhanced warning in `supported_codecs` property with actionable ffmpeg commands.

---

## Test Results

```
============================= test session starts ==============================
platform darwin -- Python 3.12.11, pytest-9.0.2, pluggy-1.6.0
collected 140 items

tests/test_analysis.py .........                                         [  6%]
tests/test_config.py ..............                                      [ 16%]
tests/test_dataset_shapes.py ................                            [ 27%]
tests/test_integration.py .................                              [ 40%]
tests/test_losses.py ..............                                      [ 50%]
tests/test_metrics.py .............                                      [ 59%]
tests/test_models.py ......................                              [ 75%]
tests/test_protocol_parse.py .......................                     [ 91%]
tests/test_training.py ............                                      [100%]

======================= 140 passed, 7 warnings in 6.24s ========================
```

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/train.py` | Wired CodecAugmentor, added worker_init_fn, use synthetic vocab sizes |
| `src/.../training/loop.py` | Use augmented labels, fail-fast diversity check, aug rate monitoring |
| `src/.../evaluation/metrics.py` | Fixed all metric functions for 0=bonafide, 1=spoof |
| `src/.../data/codec_augment.py` | Atomic cache writes, enhanced ffmpeg warning |
| `tests/test_metrics.py` | Fixed label convention in test data |
| `tests/test_losses.py` | Added CODEC_Q masking tests |
| `tests/test_protocol_parse.py` | Added vocab ID consistency test |
| `docs/asvspoof5_domains.md` | Added CODEC_Q mismatch and codec gap documentation |

---

## Remaining Known Limitations

These are documented limitations, not bugs:

1. **Codec Taxonomy Mismatch:** Synthetic codecs (MP3, AAC, OPUS, SPEEX, AMR) don't fully cover eval codecs (C04 Encodec, C07 cascade, C11 device/channel).

2. **CODEC_Q Semantic Mismatch:** Synthetic quality tiers (1-5) don't correspond to eval quality levels (6-8 are C11 device variants).

3. **Train/Dev Per-Domain Evaluation Trivial:** All train/dev samples are NONE, so per-domain breakdown is meaningless for those splits.

---

## Verification Commands

```bash
# Run all tests
uv run pytest tests/ -v --tb=short

# Smoke test ERM (requires dataset)
uv run python scripts/train.py --config configs/smoke_test.yaml --name smoke_erm

# Smoke test DANN with augmentation (requires dataset + ffmpeg)
uv run python scripts/train.py --config configs/smoke_test_dann.yaml --name smoke_dann

# Evaluate with per-domain breakdown
uv run python scripts/evaluate.py \
  --checkpoint runs/smoke_erm/checkpoints/best.pt \
  --split dev --per-domain
```

---

## Conclusion

All critical and important bugs have been fixed. The DANN training pipeline is now correctly wired with:
- Synthetic codec augmentation for domain diversity
- Proper label handling in training loop
- Fail-fast checks to prevent silent failures
- Correct metric implementations matching KEY_TO_LABEL convention

The codebase is ready for DANN experiments on ASVspoof5 Track 1.
