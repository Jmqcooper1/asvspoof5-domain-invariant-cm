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
