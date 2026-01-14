# Augmentation Design

This document explains the synthetic codec augmentation pipeline used for DANN training.

## Why Augmentation is Required

**Problem:** ASVspoof5 train and dev sets have **no codec diversity**.

| Split | CODEC | CODEC_Q | Samples |
|-------|-------|---------|---------|
| Train | 100% `"-"` (uncoded) | 100% `"-"` | 182,357 |
| Dev | 100% `"-"` (uncoded) | 100% `"-"` | 140,950 |
| Eval | 25% uncoded, 75% coded (C01-C11) | 0-8 | 680,774 |

**Consequence:** A domain discriminator trained on train/dev protocol labels learns nothing (all samples have the same domain). DANN degenerates to ERM.

**Solution:** Apply synthetic codec compression during training to create domain diversity.

---

## Synthetic Codec Set

We simulate codec compression using ffmpeg with the following configurations:

### Codec Families

| Synthetic ID | Codec | ffmpeg Encoder | Bandwidth |
|--------------|-------|----------------|-----------|
| 0 | NONE | - | Original |
| 1 | MP3 | libmp3lame | Wideband |
| 2 | AAC | aac | Wideband |
| 3 | OPUS | libopus | Wideband |
| 4 | SPEEX | libspeex | Wideband |
| 5 | AMR | libopencore_amrnb | Narrowband (8kHz) |

### Bitrate Tiers (Quality Levels 1-5)

| Quality | MP3 (kbps) | AAC (kbps) | OPUS (kbps) | SPEEX (kbps) | AMR (kbps) |
|---------|------------|------------|-------------|--------------|------------|
| 1 (lowest) | 64 | 32 | 12 | 8 | 6 |
| 2 | 96 | 64 | 24 | 16 | 9 |
| 3 | 128 | 96 | 48 | 24 | 12 |
| 4 | 192 | 128 | 64 | 32 | 18 |
| 5 (highest) | 256 | 192 | 96 | 44 | 23 |

---

## Mapping to ASVspoof5 Codec Families

| Synthetic Codec | ASVspoof5 Codec IDs | Notes |
|-----------------|---------------------|-------|
| NONE | `-` (uncoded) | Exact match |
| MP3 | C05 (mp3_wb) | Direct match |
| AAC | C06 (m4a_wb) | M4A container uses AAC |
| OPUS | C01 (opus_wb), C08 (opus_nb) | Wideband variant |
| SPEEX | C03 (speex_wb), C10 (speex_nb) | Wideband variant |
| AMR | C02 (amr_wb), C09 (amr_nb) | Narrowband variant (8kHz) |

---

## Codec Coverage Gaps

Some ASVspoof5 eval codecs are **not covered** by synthetic augmentation:

| Eval Codec | Description | Why Not Simulated | Expected Impact |
|------------|-------------|-------------------|-----------------|
| **C04** | Encodec (neural codec) | Requires neural codec model; fundamentally different artifacts | DANN may not improve |
| **C07** | MP3 + Encodec cascade | Compound degradation; only MP3 portion approximated | Partial improvement |
| **C11** | Device/channel (Bluetooth, cable, microphone) | Acoustic simulation, not codec; requires impulse responses | DANN may not improve |

---

## Augmentation Pipeline

### Configuration

```yaml
augmentation:
  enabled: true
  codec_prob: 0.5           # P(apply any codec)
  codecs: ["MP3", "AAC", "OPUS"]  # Codecs to sample from
  qualities: [1, 2, 3, 4, 5]      # Quality levels to sample from
  cache_dir: null                  # Optional caching
```

### Augmentation Flow

```
For each training sample:
    1. Load waveform from FLAC file
    2. With probability codec_prob:
        a. Sample codec uniformly from supported_codecs
        b. Sample quality uniformly from qualities
        c. Encode to codec format (ffmpeg)
        d. Decode back to 16kHz mono PCM
        e. Return (augmented_waveform, codec_id, quality_id)
    3. Else:
        Return (original_waveform, NONE, 0)
```

### Domain Label Generation

```python
# Synthetic domain vocabulary
SYNTHETIC_CODEC_VOCAB = {
    "NONE": 0, "MP3": 1, "AAC": 2, 
    "OPUS": 3, "SPEEX": 4, "AMR": 5
}

SYNTHETIC_QUALITY_VOCAB = {
    "NONE": 0, "1": 1, "2": 2, 
    "3": 3, "4": 4, "5": 5
}
```

---

## Expected Behavior

### Where DANN Should Help

| Eval Codec | Synthetic Coverage | Expected DANN Improvement |
|------------|-------------------|---------------------------|
| C01 (opus_wb) | ✅ OPUS | Likely improvement |
| C02 (amr_wb) | ✅ AMR | Likely improvement |
| C03 (speex_wb) | ✅ SPEEX | Likely improvement |
| C04 (encodec) | ❌ None | Minimal/no improvement |
| C05 (mp3_wb) | ✅ MP3 | Likely improvement |
| C06 (m4a_wb) | ✅ AAC | Likely improvement |
| C07 (mp3+encodec) | ⚠️ Partial (MP3) | Partial improvement |
| C08 (opus_nb) | ✅ OPUS | Likely improvement |
| C09 (amr_nb) | ✅ AMR | Likely improvement |
| C10 (speex_nb) | ✅ SPEEX | Likely improvement |
| C11 (device) | ❌ None | Minimal/no improvement |

### Interpreting Per-Domain Results

When analyzing per-domain EER:
1. Compare ERM vs DANN for each codec
2. Larger DANN improvement expected for covered codecs
3. Similar or worse performance expected for uncovered codecs (C04, C07, C11)
4. This validates the domain-invariance hypothesis

---

## Troubleshooting

### DANN Degenerates to ERM

**Symptoms:**
- Domain discriminator accuracy stuck at ~100%
- No domain loss gradient
- "Only 1 unique codec in batch" warning

**Causes:**
1. `augmentation.enabled: false`
2. `codec_prob: 0` (no augmentation applied)
3. ffmpeg not installed
4. No supported codecs (all encoders missing)

**Diagnostic:**
```bash
# Check ffmpeg encoders
ffmpeg -encoders 2>/dev/null | grep -E 'mp3|aac|opus'

# Run inspect_domains to verify protocol diversity
python scripts/inspect_domains.py
```

### Augmentation Too Slow

**Solutions:**
1. Enable caching: `cache_dir: /path/to/cache`
2. Reduce `num_workers` if disk I/O bottleneck
3. Pre-compute augmented files offline

---

## Implementation Reference

See `src/asvspoof5_domain_invariant_cm/data/codec_augment.py` for full implementation:
- `CodecAugmentor` class
- `SYNTHETIC_CODEC_VOCAB` and `SYNTHETIC_QUALITY_VOCAB` dictionaries
- `apply_codec_ffmpeg()` function
- Cache management logic
