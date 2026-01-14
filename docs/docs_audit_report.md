# Documentation Audit Report

**Date:** 2026-01-14  
**Scope:** Validate all markdown documentation files against actual implementation

## Executive Summary

This audit reviewed all documentation files in the repository and cross-referenced them with the actual codebase implementation. The documentation is **mostly accurate** but requires targeted updates to:
1. Explicitly state that DANN requires synthetic augmentation (train/dev have no codec diversity)
2. Clarify CODEC_Q semantic mismatch between synthetic training and eval
3. Document ffmpeg dependencies and codec coverage gaps
4. Add reproducibility and limitations documentation

---

## 1. Documentation Inventory

| Document | Purpose | Status | Changes Required |
|----------|---------|--------|------------------|
| `docs/asvspoof5_domains.md` | Domain label reality check | **VALID** | Minor additions |
| `docs/dataset.md` | Dataset setup guide | **VALID** | Add ffmpeg/caching sections |
| `docs/evaluation.md` | Evaluation metrics | **VALID** | Clarify per-domain scope |
| `docs/methodology.md` | ERM vs DANN methods | **VALID** | Add mismatch/reproducibility |
| `docs/proposal.md` | Research proposal | **VALID** | No changes (reference doc) |
| `docs/prompt.md` | AI coding prompt | **VALID** | No changes (reference doc) |
| `docs/audit_report_full.md` | Previous bug audit | **VALID** | No changes (historical) |
| `README.md` | Main usage guide | **VALID** | Add smoke test, ffmpeg, env reqs |

### New Documents Needed

| Document | Purpose |
|----------|---------|
| `docs/augmentation_design.md` | Why augmentation required, codec mapping |
| `docs/reproducibility.md` | Seeding, determinism, caching |
| `docs/limitations.md` | Known gaps and constraints |

---

## 2. Critical Consistency Check Results

### 2.1 Domain Reality Check ✅ PASS

**Requirement:** All docs must state train/dev have no codec diversity; DANN requires synthetic augmentation.

**Findings:**
- `docs/asvspoof5_domains.md` correctly states:
  - Train/dev are 100% uncoded (`CODEC="-"`, `CODEC_Q="-"`)
  - "DANN degenerates to ERM without synthetic codec augmentation"
  - Eval contains C01-C11 codec diversity

- Code matches:
  - `scripts/train.py` lines 383-405: Creates augmentor when `method == "dann"` and `augmentation.enabled`
  - `src/asvspoof5_domain_invariant_cm/training/loop.py` lines 126-135: Fail-fast check for domain diversity in first 10 batches

**Minor Gap:** `docs/methodology.md` does not explicitly state DANN domain labels are synthetic. Fixed in this PR.

### 2.2 Held-Out Codec Wording ✅ PASS (with clarification)

**Requirement:** "Held-out codec within train/dev" is misleading since train/dev have only `CODEC='-'`.

**Findings:**
- `docs/asvspoof5_domains.md` already explains this limitation
- `scripts/run_held_out_codec.py` operates on **synthetic augmentation domains**, not protocol domains

**Recommendation:** Added clarification in `docs/methodology.md` that held-out codec analysis uses synthetic domains.

### 2.3 CODEC_Q Interpretation ✅ PASS

**Requirement:** CODEC_Q is not globally ordinal; eval 6-8 are C11 device variants.

**Findings:**
- `docs/asvspoof5_domains.md` lines 113-128 correctly document:
  - Synthetic quality levels 1-5 don't correspond to eval quality levels
  - Eval quality levels 6-8 are C11 device variants (Bluetooth, cable, MST)
  - "Per-CODEC_Q evaluation on the eval set is purely analytical"

**Status:** Already correct in documentation.

### 2.4 Metrics Polarity / Label Convention ✅ PASS

**Requirement:** Docs and code must agree on label encoding and score direction.

**Code conventions (verified):**
- `KEY_TO_LABEL = {"bonafide": 0, "spoof": 1}` in `src/asvspoof5_domain_invariant_cm/data/asvspoof5.py`
- Score direction: higher = more likely bonafide (P(bonafide) from softmax)
- `src/asvspoof5_domain_invariant_cm/evaluation/metrics.py` correctly implements this

**Documentation:**
- `README.md` line 631-632: "**bonafide = 0, spoof = 1**" and "Score convention: **higher score = more likely bonafide**"
- `docs/evaluation.md` line 36: Code comment shows `labels: 1 for bonafide, 0 for spoof` - **INCORRECT**

**Fix Applied:** Updated `docs/evaluation.md` to match actual convention.

---

## 3. Document-by-Document Changes

### 3.1 `docs/asvspoof5_domains.md`

**Already Contains:**
- ✅ Exact counts/unique values per split
- ✅ DANN requires augmentation implication
- ✅ CODEC_Q semantic mismatch section
- ✅ Codec coverage gaps (C04/C07/C11)
- ✅ Normalization note

**Added:**
- "How We Operationalize Domains for Training" section with:
  - Synthetic codec families used (MP3, AAC, OPUS, SPEEX, AMR)
  - Synthetic quality tiers (1-5)
  - Mapping to ASVspoof codec IDs

### 3.2 `docs/dataset.md`

**Already Contains:**
- ✅ Correct directory names (`flac_T`, `flac_D`, `flac_E_eval`)
- ✅ Protocols path (`ASVspoof5_protocols/`)
- ✅ Protocol parsing (whitespace-separated)
- ✅ Which tar files to download
- ✅ `ASVSPOOF5_ROOT` env var

**Added:**
- "FFmpeg Dependency" section:
  - Required encoders for MP3/AAC/OPUS
  - Optional encoders for SPEEX/AMR
  - How to check (`ffmpeg -encoders`)
- "Synthetic Augmentation Caching" section:
  - Cache directory config
  - Expected disk usage
  - Atomic write behavior

### 3.3 `docs/evaluation.md`

**Already Contains:**
- ✅ Primary metrics (minDCF, EER)
- ✅ Per-domain breakdown by CODEC/CODEC_Q
- ✅ Domain probe accuracy

**Fixed:**
- Label encoding in code example (was backwards)

**Added:**
- "Per-Domain Reporting on Eval Only" emphasis
- Note on domain normalization (`'-'` and `'0'` → `NONE`)

### 3.4 `docs/methodology.md`

**Already Contains:**
- ✅ ERM vs DANN definitions
- ✅ Model architecture diagram
- ✅ Gradient reversal layer
- ✅ Multi-head discriminator

**Added:**
- "Domain Labels During Training vs Evaluation" section clarifying:
  - Training uses synthetic domain labels from augmentation
  - Evaluation uses protocol metadata for analysis
- "Known Domain Mismatch" section:
  - Synthetic codecs vs ASVspoof codec IDs
  - Limitations (Encodec/C07/C11 not simulated)
- "Reproducibility Notes" section:
  - Worker seeding for augmentation
  - Deterministic eval cropping

### 3.5 `README.md`

**Already Contains:**
- ✅ All required CLI commands
- ✅ Dataset setup instructions
- ✅ Training/evaluation commands
- ✅ Analysis commands (probes/CKA/patching)
- ✅ Baselines (TRILLsson, LFCC-GMM)
- ✅ Label/score conventions

**Added:**
- "Smoke Test" command section
- "Environment Requirements" section with:
  - FFmpeg requirement
  - Python version
  - GPU notes

---

## 4. New Documents Created

### 4.1 `docs/augmentation_design.md`

Contents:
- Why augmentation is required (train/dev domain constant)
- Synthetic codec set and bitrate tiers
- Mapping to ASVspoof codec families
- Limitations and expected behavior

### 4.2 `docs/reproducibility.md`

Contents:
- Seeding policy (torch/numpy/random + worker_init_fn)
- Deterministic eval (center crop vs random crop)
- Caching determinism
- Multi-seed experiments

### 4.3 `docs/limitations.md`

Contents:
- Domain mismatch (Encodec/C07/C11)
- CODEC_Q semantics mismatch
- FFmpeg encoder availability
- Computational considerations

---

## 5. Files Modified

| File | Changes |
|------|---------|
| `docs/asvspoof5_domains.md` | Added "How We Operationalize Domains" section |
| `docs/dataset.md` | Added ffmpeg and caching sections |
| `docs/evaluation.md` | Fixed label convention, added per-domain scope |
| `docs/methodology.md` | Added domain mismatch and reproducibility |
| `README.md` | Added smoke test, env requirements |
| `docs/augmentation_design.md` | New file |
| `docs/reproducibility.md` | New file |
| `docs/limitations.md` | New file |

---

## 6. Verification Checklist

- [x] Train/dev no codec diversity stated
- [x] DANN requires synthetic augmentation stated
- [x] Eval contains C01-C11 diversity stated
- [x] Uncoded normalization documented
- [x] CODEC_Q not globally ordinal stated
- [x] CODEC_Q 6-8 = C11 devices stated
- [x] Label encoding consistent (bonafide=0, spoof=1)
- [x] Score direction consistent (higher=bonafide)
- [x] FFmpeg dependency documented
- [x] Codec coverage gaps documented
- [x] Seeding/reproducibility documented
