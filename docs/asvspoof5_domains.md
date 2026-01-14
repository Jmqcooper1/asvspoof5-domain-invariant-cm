# ASVspoof5 Domain Labels: Reality Check

This document describes the actual CODEC and CODEC_Q domain labels in the ASVspoof5 Track 1 protocol files, based on direct inspection of the data.

## Summary

| Split | Samples | CODEC values | CODEC_Q values |
|-------|---------|--------------|----------------|
| Train | 182,357 | 100% "-" (uncoded) | 100% "-" |
| Dev | 140,950 | 100% "-" (uncoded) | 100% "-" |
| Eval | 680,774 | "-" (~171k) + C01-C11 (~509k) | "0" (~171k) + "1-8" (~509k) |

## Critical Finding: No Codec Diversity in Train/Dev

**Train and dev sets have NO codec diversity.** All samples are uncoded (CODEC="-", CODEC_Q="-").

This means:
- A domain discriminator trained on train/dev labels learns nothing (constant labels)
- Gradient reversal produces zero useful signal
- **DANN degenerates to ERM without synthetic codec augmentation**

## Eval Set Domain Distribution

### CODEC Distribution (Eval)

| CODEC | Count | Description |
|-------|-------|-------------|
| `-` | 171,602 | Uncoded/original |
| C01 | 47,253 | opus_wb |
| C02 | 47,426 | amr_wb |
| C03 | 47,987 | speex_wb |
| C04 | 47,699 | encodec_wb |
| C05 | 47,541 | mp3_wb |
| C06 | 41,554 | m4a_wb |
| C07 | 47,348 | mp3_encodec_wb |
| C08 | 46,678 | opus_nb |
| C09 | 41,599 | amr_nb |
| C10 | 47,477 | speex_nb |
| C11 | 46,610 | Device/channel (MST, bluetooth, cable) |

### CODEC_Q Distribution (Eval)

| CODEC_Q | Count | Notes |
|---------|-------|-------|
| 0 | 171,602 | Uncoded samples (CODEC="-") |
| 1 | 98,375 | Lowest quality/bitrate |
| 2 | 98,339 | |
| 3 | 98,280 | |
| 4 | 98,346 | |
| 5 | 98,372 | Highest quality/bitrate |
| 6 | 5,815 | C11 device variants only |
| 7 | 5,816 | C11 device variants only |
| 8 | 5,829 | C11 device variants only |

## Normalization Requirements

The protocol files use different conventions for "uncoded" across splits:

| Split | CODEC uncoded | CODEC_Q uncoded |
|-------|---------------|-----------------|
| Train | "-" | "-" |
| Dev | "-" | "-" |
| Eval | "-" | "0" |

**Both "-" and "0" must normalize to "NONE"** for consistent domain labels.

## Codec Config Reference

From `ASVspoof5.codec.config.csv`, the codec families are:

| ID | Codec Family | Bandwidth | Quality Range |
|----|--------------|-----------|---------------|
| C01 | opus_wb | Wideband | 6-30 kbps |
| C02 | amr_wb | Wideband | 6.6-23 kbps |
| C03 | speex_wb | Wideband | 5.75-34.2 kbps |
| C04 | encodec_wb | Wideband | 1.5-24 kbps |
| C05 | mp3_wb | Wideband | 45-260 kbps |
| C06 | m4a_wb | Wideband | 16-128 kbps |
| C07 | mp3_encodec_wb | Wideband | Cascaded |
| C08 | opus_nb | Narrowband | 4-20 kbps |
| C09 | amr_nb | Narrowband | 4.75-12.2 kbps |
| C10 | speex_nb | Narrowband | 3.95-24.6 kbps |
| C11 | Device/Channel | Various | BT/cable/MST |

## Supervisor Question: C00-C11 vs C01-C11

Charlotte asked about C00-C11 from Table 3 of arXiv:2502.08857.

**Answer:** The actual protocol data uses **C01-C11** (not C00-C11). There is no C00 in the data. Uncoded samples use "-" for CODEC.

## Implications for DANN Training

To make domain-adversarial training meaningful, we need **synthetic codec augmentation**:

1. Apply codec compression to clean training audio
2. Create diverse domain labels from augmentation
3. Train domain discriminator on synthetic domains
4. Map synthetic domains to approximate real codec families

Target synthetic domains:
- `NONE` - clean/original (no augmentation)
- `MP3` - maps to C05, C07 families
- `AAC` - maps to C06 family
- `OPUS` - maps to C01, C08 families
- `SPEEX` - maps to C03, C10 families
- `AMR` - maps to C02, C09 families

## Leakage Control

When creating any custom splits (e.g., held-out codec analysis), group by `CODEC_SEED` to ensure coded variants of the same original utterance stay together.

## CODEC_Q Semantic Mismatch (Important Limitation)

The CODEC_Q domain has different semantics between synthetic augmentation and eval protocol:

**Synthetic augmentation quality levels:** 1-5 (arbitrary bitrate tiers per codec)

**Eval protocol quality levels:** 0-8, where:
- 0: Uncoded (CODEC="-")
- 1-5: Codec bitrate tiers (per-codec, not globally comparable)
- 6-8: C11 device/channel variants (Bluetooth, cable, MST)

**Implication:** The CODEC_Q adversarial head trains on synthetic quality tiers
(1-5) which do NOT directly correspond to eval quality levels. Quality levels
are categorical (codec-specific), not ordinal across codec families.

Per-CODEC_Q evaluation on the eval set is purely analytical - the model has
never seen quality levels 6-8 during training (C11 device variants).

## Codec Coverage Gaps (Important Limitation)

**Synthetic augmentation covers:** MP3, AAC, OPUS, SPEEX, AMR

**Eval codecs NOT covered by synthetic augmentation:**

| Eval Codec | Description | Why Not Simulated |
|------------|-------------|-------------------|
| **C04 (Encodec)** | Neural codec | Fundamentally different artifacts; no ffmpeg encoder |
| **C07 (MP3+Encodec)** | Cascaded codecs | Compound degradation not easily simulatable |
| **C11 (Device/channel)** | Bluetooth, cable, microphone | Requires acoustic simulation, not codec |

**Expected behavior:** DANN may NOT generalize to C04, C07, C11 domains since
they were never seen during training. Per-domain evaluation should reveal
where domain-invariant gains occur vs. where they don't.

**Potential mitigations (out of scope for current work):**
- For C11: Add channel simulation (bandwidth limit + noise + reverb)
- For C04/C07: Would require neural codec implementation or pre-coded samples
