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
