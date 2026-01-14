# ASVspoof 5 Dataset Setup (Track 1)

This document describes how to download, unpack, and parse the ASVspoof 5 dataset for Track 1 (stand-alone bonafide vs spoof detection).

## Official Sources

- **Official site:** https://www.asvspoof.org/
- **Zenodo record:** https://zenodo.org/records/14498691
- **HuggingFace mirror:** https://huggingface.co/datasets/jungjee/asvspoof5

## What to Download

### For Initial Development (Fast Iteration)

Download these first to validate your pipeline:

1. `ASVspoof5_protocols.tar.gz` - All protocol/label TSV files
2. `flac_T_aa.tar` - Training audio subset
3. `flac_D_aa.tar` - Dev audio subset

### For Full Experiments

**Training audio:**
- `flac_T_aa.tar` through `flac_T_ae.tar`

**Dev audio:**
- `flac_D_aa.tar` through `flac_D_ac.tar`

**Evaluation audio (download last, after models are stable):**
- `flac_E_aa.tar` through `flac_E_aj.tar`

## Directory Structure After Unpacking

```
data/raw/asvspoof5/
├── ASVspoof5_protocols/
│   ├── ASVspoof5.train.tsv
│   ├── ASVspoof5.dev.track_1.tsv
│   ├── ASVspoof5.eval.track_1.tsv
│   └── ASVspoof5.codec.config.csv
├── flac_T/
│   └── *.flac
├── flac_D/
│   └── *.flac
└── flac_E_eval/
    └── *.flac
```

## Protocol File Format

**Important:** Although files use `.tsv` extension, they are **whitespace-separated** (not tab-separated). Parse by splitting on any whitespace.

### Track 1 Columns

| Column | Description |
|--------|-------------|
| `SPEAKER_ID` | Speaker identifier |
| `FLAC_FILE_NAME` | Audio filename (without path) |
| `SPEAKER_GENDER` | M/F |
| `CODEC` | Codec type/family |
| `CODEC_Q` | Codec quality setting |
| `CODEC_SEED` | Seed linking coded variants to originals |
| `ATTACK_TAG` | Attack algorithm tag |
| `ATTACK_LABEL` | Attack label (e.g., A01, A02, ...) |
| `KEY` | bonafide / spoof |
| `TMP` | Temporary/reserved field |

### Parsing Example (Python)

```python
import pandas as pd

def load_protocol(path: str) -> pd.DataFrame:
    """Load ASVspoof 5 protocol file (whitespace-separated)."""
    columns = [
        "speaker_id", "flac_file", "gender", "codec", "codec_q",
        "codec_seed", "attack_tag", "attack_label", "key", "tmp"
    ]
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype=str,
    )
    return df
```

## Domain Labels for DANN

For domain-adversarial training, we use:

- **CODEC domain:** `codec` column values
- **CODEC_Q domain:** `codec_q` column values

The multi-head discriminator predicts both separately.

## Leakage Prevention via CODEC_SEED

When creating custom train/test splits (e.g., leave-one-codec-out):

1. Group samples by `codec_seed`
2. Ensure all variants of the same seed stay in the same split
3. This prevents near-duplicate leakage between splits

```python
def get_codec_seed_groups(df: pd.DataFrame) -> dict:
    """Group samples by codec_seed for split stratification."""
    return df.groupby("codec_seed").indices
```

## Audio Format

- Format: FLAC (lossless)
- Sample rate: 16 kHz
- Channels: Mono

## Evaluation Metrics

Track 1 uses:
- **Primary:** minDCF (minimum Detection Cost Function)
- **Secondary:** EER (Equal Error Rate)
- **Optional:** Cllr, actDCF (calibration-aware)

Use the official ASVspoof evaluation package for consistent scoring.

---

## FFmpeg Dependency (for DANN Training)

DANN training requires synthetic codec augmentation, which uses ffmpeg.

### Required Encoders

| Codec | ffmpeg Encoder | Package/Library |
|-------|----------------|-----------------|
| MP3 | libmp3lame | LAME (usually included) |
| AAC | aac | Built-in (or libfdk_aac) |
| OPUS | libopus | libopus |

### Optional Encoders

| Codec | ffmpeg Encoder | Package/Library |
|-------|----------------|-----------------|
| SPEEX | libspeex | libspeex (often not included) |
| AMR | libopencore_amrnb | opencore-amr |

### Check Available Encoders

```bash
# List all available audio encoders
ffmpeg -encoders 2>/dev/null | grep -E 'mp3|aac|opus|speex|amr'

# Expected output for basic support:
#  A..... aac              AAC (Advanced Audio Coding)
#  A..... libmp3lame       libmp3lame MP3 (MPEG audio layer 3)
#  A..... libopus          libopus Opus
```

### Installing Full Codec Support

**macOS (Homebrew):**
```bash
brew install ffmpeg --with-libopus --with-speex
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg libopus-dev libspeex-dev libopencore-amrnb-dev
```

**Note:** DANN will work with just MP3/AAC/OPUS. SPEEX and AMR are optional and provide additional domain diversity if available.

---

## Synthetic Augmentation Caching

The codec augmentor can cache augmented audio to disk for faster subsequent epochs.

### Configuration

```yaml
augmentation:
  enabled: true
  codec_prob: 0.5
  codecs: ["MP3", "AAC", "OPUS"]
  qualities: [1, 2, 3, 4, 5]
  cache_dir: /path/to/cache  # Optional: set to cache augmented files
```

### Cache Behavior

- **Cache key:** MD5 hash of original path + codec + quality
- **Cache format:** FLAC files (lossless storage of augmented audio)
- **Atomic writes:** Uses temp file + `os.replace()` to prevent corruption
- **Cache hit:** Returns cached audio without re-encoding

### Expected Disk Usage

| Dataset Split | Approx. Samples | Cache Size (all codecs/qualities) |
|---------------|-----------------|-----------------------------------|
| Train | 182,357 | ~50-100 GB (depending on codec_prob) |
| Dev | 140,950 | N/A (augmentation only during training) |

### Recommendation

For initial experiments, disable caching (`cache_dir: null`) to avoid disk usage. Enable caching for multi-epoch training runs to speed up data loading.

---

## Domain Label Normalization

The protocol files use different conventions for "uncoded" samples:

| Split | CODEC uncoded | CODEC_Q uncoded | Normalized to |
|-------|---------------|-----------------|---------------|
| Train | `"-"` | `"-"` | `"NONE"` |
| Dev | `"-"` | `"-"` | `"NONE"` |
| Eval | `"-"` | `"0"` | `"NONE"` |

The `normalize_domain_value()` function handles this:
- `"-"` always maps to `"NONE"`
- `"0"` maps to `"NONE"` only for CODEC_Q (not CODEC)
