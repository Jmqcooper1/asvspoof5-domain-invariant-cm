#!/usr/bin/env python3
"""Frozen-backbone domain probe: can SSL features distinguish codecs?

This is a PRE-TRAINING diagnostic. It answers a critical question:
    "Do frozen WavLM/Wav2Vec2 representations contain information that
     distinguishes codec-augmented audio from originals?"

If a simple linear probe CANNOT distinguish codecs from frozen features,
then DANN with a frozen backbone is fundamentally broken — the gradient
reversal layer has nothing to work with.

Unlike probe_domain.py (which requires a trained checkpoint), this script
operates directly on frozen backbone outputs with NO training involved.

Usage:
    # Basic usage with WavLM (needs ASVSPOOF5_ROOT set)
    python scripts/run_domain_probe.py --backbone wavlm --num-samples 5000

    # Use Wav2Vec2 instead
    python scripts/run_domain_probe.py --backbone w2v2 --num-samples 5000

    # Use synthetic audio (no dataset needed — for quick sanity checks)
    python scripts/run_domain_probe.py --backbone wavlm --synthetic --num-samples 200

    # Specify output location
    python scripts/run_domain_probe.py --backbone wavlm --output-dir results/probe_wavlm

Interpretation:
    Binary accuracy > 60% on ANY layer → codec info IS present, DANN can work
    Binary accuracy ≈ 50% on ALL layers → codec info NOT present, frozen DANN broken
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKBONES = {
    "wavlm": {
        "class": "WavLMModel",
        "module": "transformers",
        "pretrained": "microsoft/wavlm-base-plus",
        "num_layers": 12,
        "hidden_size": 768,
    },
    "w2v2": {
        "class": "Wav2Vec2Model",
        "module": "transformers",
        "pretrained": "facebook/wav2vec2-base",
        "num_layers": 12,
        "hidden_size": 768,
    },
}

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 4.0  # shorter than training for speed
MAX_SAMPLES = int(MAX_DURATION_SEC * SAMPLE_RATE)

# Codec configs: (codec_name, ffmpeg_encoder, format, extension, bitrates_kbps)
CODEC_CONFIGS = {
    "MP3":  ("libmp3lame", "mp3",  ".mp3",  [64, 128, 256]),
    "AAC":  ("aac",        "adts", ".aac",  [32, 96, 192]),
    "OPUS": ("libopus",    "opus", ".opus", [12, 48, 96]),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen-backbone domain probe for codec distinguishability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--backbone", choices=list(BACKBONES), default="wavlm",
        help="SSL backbone to probe (default: wavlm)",
    )
    p.add_argument(
        "--num-samples", type=int, default=5000,
        help="Number of audio samples to use (default: 5000)",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic audio instead of ASVspoof5 data (for quick tests)",
    )
    p.add_argument(
        "--split", choices=["train", "dev"], default="train",
        help="ASVspoof5 split to load (default: train)",
    )
    p.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for feature extraction (default: 8, keep small for CPU)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: results/domain_probe_{backbone})",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------
def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


def check_codec_support() -> dict[str, bool]:
    """Check which codecs ffmpeg supports."""
    supported = {}
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10,
        )
        for name, (encoder, *_) in CODEC_CONFIGS.items():
            supported[name] = encoder in result.stdout
    except Exception:
        for name in CODEC_CONFIGS:
            supported[name] = False
    return supported


def apply_codec(
    waveform: np.ndarray,
    codec_name: str,
    bitrate_kbps: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray | None:
    """Apply codec compression via ffmpeg and return decoded audio.

    Args:
        waveform: float32 numpy array of shape (num_samples,).
        codec_name: One of MP3, AAC, OPUS.
        bitrate_kbps: Target bitrate.
        sample_rate: Audio sample rate.

    Returns:
        Decoded waveform as float32 numpy array, or None on failure.
    """
    import soundfile as sf

    encoder, fmt, ext, _ = CODEC_CONFIGS[codec_name]
    tmp_in = tmp_enc = tmp_out = None

    try:
        # Write input WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_in = f.name
        sf.write(tmp_in, waveform, sample_rate)

        # Encode
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            tmp_enc = f.name
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_in,
                "-ar", str(sample_rate), "-ac", "1",
                "-c:a", encoder, "-b:a", f"{bitrate_kbps}k",
                "-f", fmt, tmp_enc,
            ],
            capture_output=True, check=True, timeout=30,
        )

        # Decode back to WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_out = f.name
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_enc,
                "-ar", str(sample_rate), "-ac", "1", tmp_out,
            ],
            capture_output=True, check=True, timeout=30,
        )

        data, _ = sf.read(tmp_out, dtype="float32")
        return data

    except Exception as e:
        logger.debug(f"Codec {codec_name}@{bitrate_kbps}k failed: {e}")
        return None

    finally:
        for p in (tmp_in, tmp_enc, tmp_out):
            if p and os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_real_audio(split: str, num_samples: int, seed: int) -> list[np.ndarray]:
    """Load real ASVspoof5 audio samples.

    Requires ASVSPOOF5_ROOT to be set and manifests to exist.
    """
    import soundfile as sf

    # Try to use the project's path utilities
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from asvspoof5_domain_invariant_cm.utils.paths import get_manifest_path
        manifest_path = get_manifest_path(split)
    except Exception:
        # Fallback: look for manifest in standard location
        project_root = Path(__file__).resolve().parent.parent
        manifest_path = project_root / "data" / "manifests" / f"{split}.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run 'python scripts/prepare_asvspoof5.py' first, or use --synthetic."
        )

    import pandas as pd
    df = pd.read_parquet(manifest_path)

    rng = np.random.RandomState(seed)
    n = min(num_samples, len(df))
    indices = rng.choice(len(df), size=n, replace=False)

    waveforms = []
    for idx in indices:
        audio_path = df.iloc[idx]["audio_path"]
        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            # Crop/pad to fixed length
            if len(data) > MAX_SAMPLES:
                start = rng.randint(0, len(data) - MAX_SAMPLES)
                data = data[start:start + MAX_SAMPLES]
            elif len(data) < MAX_SAMPLES:
                data = np.pad(data, (0, MAX_SAMPLES - len(data)))
            waveforms.append(data)
        except Exception as e:
            logger.debug(f"Failed to load {audio_path}: {e}")
            continue

    if len(waveforms) == 0:
        raise RuntimeError("Failed to load any audio samples. Check ASVSPOOF5_ROOT.")

    logger.info(f"Loaded {len(waveforms)} audio samples from {split} split")
    return waveforms


def generate_synthetic_audio(num_samples: int, seed: int) -> list[np.ndarray]:
    """Generate diverse synthetic audio for testing.

    Creates a mix of:
    - Speech-like signals (multiple formants with noise)
    - Tonal signals with harmonics
    - Noise bursts

    This ensures the probe test is meaningful even without real data.
    """
    rng = np.random.RandomState(seed)
    waveforms = []
    t = np.linspace(0, MAX_DURATION_SEC, MAX_SAMPLES, dtype=np.float32)

    for i in range(num_samples):
        # Mix of formant-like frequencies + noise (vaguely speech-like)
        f0 = rng.uniform(80, 300)  # fundamental
        signal = np.zeros(MAX_SAMPLES, dtype=np.float32)

        # Add harmonics
        for h in range(1, rng.randint(4, 12)):
            amp = rng.uniform(0.05, 0.3) / h
            signal += np.float32(amp) * np.sin(np.float32(2 * np.pi * f0 * h) * t + np.float32(rng.uniform(0, 2 * np.pi)))

        # Add formant peaks (speech-like resonances)
        for _ in range(rng.randint(2, 5)):
            formant_f = rng.uniform(300, 4000)
            formant_bw = rng.uniform(50, 200)
            amp = rng.uniform(0.02, 0.15)
            signal += np.float32(amp) * np.sin(np.float32(2 * np.pi * formant_f) * t)

        # Add some noise
        noise_level = rng.uniform(0.001, 0.02)
        signal += noise_level * rng.randn(MAX_SAMPLES).astype(np.float32)

        # Random amplitude envelope (simulate natural dynamics)
        envelope = np.ones(MAX_SAMPLES, dtype=np.float32)
        n_segments = rng.randint(3, 8)
        segment_len = MAX_SAMPLES // n_segments
        for seg in range(n_segments):
            start = seg * segment_len
            end = min(start + segment_len, MAX_SAMPLES)
            envelope[start:end] = rng.uniform(0.2, 1.0)
        signal *= envelope

        # Normalize
        peak = np.abs(signal).max()
        if peak > 0:
            signal = signal / peak * rng.uniform(0.3, 0.9)

        waveforms.append(signal)

    logger.info(f"Generated {num_samples} synthetic audio samples")
    return waveforms


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def load_backbone(name: str) -> tuple[torch.nn.Module, dict]:
    """Load a frozen SSL backbone.

    Returns:
        (model, config_dict) with model in eval mode on CPU.
    """
    cfg = BACKBONES[name]
    logger.info(f"Loading {name} backbone: {cfg['pretrained']}")

    if name == "wavlm":
        from transformers import WavLMModel
        model = WavLMModel.from_pretrained(
            cfg["pretrained"], output_hidden_states=True,
        )
    elif name == "w2v2":
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(
            cfg["pretrained"], output_hidden_states=True,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        f"  Loaded: {cfg['num_layers']} layers, "
        f"hidden_size={cfg['hidden_size']}, frozen=True"
    )
    return model, cfg


@torch.no_grad()
def extract_all_layer_features(
    model: torch.nn.Module,
    waveforms: list[np.ndarray],
    batch_size: int = 8,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled features from every transformer layer.

    Args:
        model: HuggingFace SSL model with output_hidden_states=True.
        waveforms: List of float32 numpy arrays (each shape [T]).
        batch_size: Batch size for inference.

    Returns:
        Dict mapping layer_index (0..11) -> np.ndarray of shape (N, D).
    """
    num_layers = model.config.num_hidden_layers
    layer_features: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}

    n_batches = (len(waveforms) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(waveforms))
        batch_np = waveforms[start:end]

        # Stack into tensor
        batch_tensor = torch.tensor(np.stack(batch_np), dtype=torch.float32)

        outputs = model(batch_tensor)

        # hidden_states: tuple of (num_layers+1) tensors, each [B, T', D]
        # Index 0 is the CNN feature extractor output, 1..12 are transformer layers
        hidden_states = outputs.hidden_states

        for layer_idx in range(num_layers):
            # +1 to skip CNN feature extractor output
            hs = hidden_states[layer_idx + 1]  # [B, T', D]
            pooled = hs.mean(dim=1).numpy()     # [B, D]
            layer_features[layer_idx].append(pooled)

        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
            logger.info(
                f"  Feature extraction: batch {batch_idx+1}/{n_batches} "
                f"({(batch_idx+1)*100//n_batches}%)"
            )

    # Concatenate
    result = {}
    for layer_idx in range(num_layers):
        result[layer_idx] = np.concatenate(layer_features[layer_idx], axis=0)

    return result


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------
def create_augmented_dataset(
    waveforms: list[np.ndarray],
    supported_codecs: dict[str, bool],
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Create dataset of original + codec-augmented audio.

    For each original waveform, creates one augmented version with a random
    codec and bitrate. The original is also included as NONE.

    Args:
        waveforms: Original audio samples.
        supported_codecs: Dict of codec_name -> is_supported.
        seed: Random seed.

    Returns:
        (all_waveforms, binary_labels, multiclass_labels, codec_names)
        binary: 0=NONE, 1=codec
        multiclass: 0=NONE, 1=MP3, 2=AAC, 3=OPUS
    """
    codecs_available = [c for c, ok in supported_codecs.items() if ok]
    if not codecs_available:
        raise RuntimeError(
            "No codecs available! Install ffmpeg with codec support:\n"
            "  apt-get install ffmpeg  # or brew install ffmpeg"
        )

    multiclass_map = {"NONE": 0, "MP3": 1, "AAC": 2, "OPUS": 3}

    rng = np.random.RandomState(seed)
    all_waveforms = []
    binary_labels = []
    multiclass_labels = []
    codec_names = []

    augmented_count = 0
    failed_count = 0

    for i, wav in enumerate(waveforms):
        # Add original
        all_waveforms.append(wav)
        binary_labels.append(0)
        multiclass_labels.append(0)
        codec_names.append("NONE")

        # Create codec-augmented version
        codec = rng.choice(codecs_available)
        _, _, _, bitrates = CODEC_CONFIGS[codec]
        bitrate = rng.choice(bitrates)

        augmented = apply_codec(wav, codec, bitrate)

        if augmented is not None:
            # Ensure same length as original
            if len(augmented) > MAX_SAMPLES:
                augmented = augmented[:MAX_SAMPLES]
            elif len(augmented) < MAX_SAMPLES:
                augmented = np.pad(augmented, (0, MAX_SAMPLES - len(augmented)))

            all_waveforms.append(augmented)
            binary_labels.append(1)
            multiclass_labels.append(multiclass_map[codec])
            codec_names.append(codec)
            augmented_count += 1
        else:
            failed_count += 1

        if (i + 1) % max(1, len(waveforms) // 10) == 0:
            logger.info(
                f"  Augmentation: {i+1}/{len(waveforms)} "
                f"(success={augmented_count}, failed={failed_count})"
            )

    logger.info(
        f"Dataset created: {len(all_waveforms)} samples "
        f"({len(waveforms)} original + {augmented_count} augmented, "
        f"{failed_count} failed)"
    )

    return (
        all_waveforms,
        np.array(binary_labels, dtype=np.int64),
        np.array(multiclass_labels, dtype=np.int64),
        codec_names,
    )


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train a logistic regression probe and return metrics.

    Args:
        X: Features of shape (N, D).
        y: Labels of shape (N,).
        test_size: Fraction held out for testing.
        seed: Random seed.

    Returns:
        Dict with accuracy, std, n_classes, class_distribution, etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)

    if n_classes < 2:
        return {
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "n_classes": n_classes,
            "status": "skipped_single_class",
        }

    # If any class has very few samples, skip
    if counts.min() < 4:
        return {
            "accuracy": float("nan"),
            "accuracy_std": float("nan"),
            "n_classes": n_classes,
            "status": "skipped_too_few_samples",
            "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        }

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use multiple stratified splits for robust estimate
    n_splits = 5
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    scores = []
    for train_idx, test_idx in sss.split(X_scaled, y):
        clf = LogisticRegression(
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
            C=1.0,
        )
        clf.fit(X_scaled[train_idx], y[train_idx])
        score = clf.score(X_scaled[test_idx], y[test_idx])
        scores.append(score)

    return {
        "accuracy": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
        "n_classes": n_classes,
        "n_splits": n_splits,
        "scores": [float(s) for s in scores],
        "class_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def print_results_table(
    results: dict[int, dict],
    num_layers: int,
    backbone_name: str,
) -> tuple[float, float]:
    """Print a formatted results table and return best accuracies.

    Returns:
        (best_binary_acc, best_multiclass_acc)
    """
    binary_chance = 0.5
    # Compute multiclass chance from actual distribution
    mc_example = results.get(0, {}).get("multiclass", {})
    mc_counts = mc_example.get("class_counts", {})
    if mc_counts:
        total = sum(mc_counts.values())
        multiclass_chance = max(mc_counts.values()) / total if total > 0 else 0.25
    else:
        multiclass_chance = 0.25

    header = f"{'Layer':>6} | {'Binary Acc':>11} | {'Multi Acc':>11} | {'Bin Chance':>10} | {'MC Chance':>10}"
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  FROZEN {backbone_name.upper()} BACKBONE — DOMAIN PROBE RESULTS")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    best_binary = 0.0
    best_binary_layer = -1
    best_multi = 0.0
    best_multi_layer = -1

    for layer in range(num_layers):
        layer_result = results.get(layer, {})
        bin_res = layer_result.get("binary", {})
        mc_res = layer_result.get("multiclass", {})

        bin_acc = bin_res.get("accuracy", float("nan"))
        mc_acc = mc_res.get("accuracy", float("nan"))

        bin_str = f"{bin_acc:.4f}" if np.isfinite(bin_acc) else "  N/A  "
        mc_str = f"{mc_acc:.4f}" if np.isfinite(mc_acc) else "  N/A  "

        # Track best
        if np.isfinite(bin_acc) and bin_acc > best_binary:
            best_binary = bin_acc
            best_binary_layer = layer
        if np.isfinite(mc_acc) and mc_acc > best_multi:
            best_multi = mc_acc
            best_multi_layer = layer

        # Highlight significant results
        marker = ""
        if np.isfinite(bin_acc) and bin_acc > 0.60:
            marker = " ★"

        print(
            f"  L{layer:02d}  | {bin_str:>11} | {mc_str:>11} | "
            f"{binary_chance:>10.4f} | {multiclass_chance:>10.4f}{marker}"
        )

    print(sep)
    print(f"  Best binary:     {best_binary:.4f} (layer {best_binary_layer})")
    print(f"  Best multiclass: {best_multi:.4f} (layer {best_multi_layer})")
    print()

    # Interpretation
    print("=" * len(header))
    print("  INTERPRETATION")
    print("=" * len(header))

    if best_binary > 0.70:
        print("  ✅ STRONG codec signal detected in frozen backbone features.")
        print("     DANN domain discriminator SHOULD be able to learn from these.")
        print("     If DANN disc is stuck at 50%, the issue is likely:")
        print("       - Gradient reversal strength (lambda) too high/low")
        print("       - Discriminator architecture too weak")
        print("       - Bug in the GRL implementation")
    elif best_binary > 0.60:
        print("  ⚠️  WEAK codec signal detected in frozen backbone features.")
        print("     DANN may work but discriminator needs to be well-tuned.")
        print("     Consider: unfreezing top backbone layers, stronger augmentation.")
    elif best_binary > 0.55:
        print("  ⚠️  MARGINAL codec signal. Barely above chance.")
        print("     DANN with frozen backbone is unlikely to work well.")
        print("     Strong recommendation: unfreeze backbone or use different features.")
    else:
        print("  ❌ NO codec signal in frozen backbone features.")
        print("     DANN with frozen backbone CANNOT work — nothing to adversarially remove.")
        print("     The domain discriminator SHOULD be at ~50% (it's correct behavior).")
        print("     To fix: UNFREEZE the backbone so DANN can learn domain-invariant features.")
    print()

    return best_binary, best_multi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output directory
    if args.output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        args.output_dir = project_root / "results" / f"domain_probe_{args.backbone}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Frozen Backbone Domain Probe")
    logger.info("=" * 60)
    logger.info(f"Backbone:    {args.backbone}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Data source: {'synthetic' if args.synthetic else f'ASVspoof5 {args.split}'}")
    logger.info(f"Output:      {args.output_dir}")
    logger.info(f"Device:      CPU (frozen backbone, no GPU needed)")

    # ---- Step 0: Check ffmpeg ----
    if not check_ffmpeg():
        logger.error(
            "❌ ffmpeg is NOT installed or not in PATH.\n"
            "   This script needs ffmpeg for codec augmentation.\n"
            "   Install it:\n"
            "     Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "     macOS:         brew install ffmpeg\n"
            "     Conda:         conda install -c conda-forge ffmpeg"
        )
        return 1

    supported_codecs = check_codec_support()
    available = [c for c, ok in supported_codecs.items() if ok]
    unavailable = [c for c, ok in supported_codecs.items() if not ok]

    logger.info(f"Codecs available: {available}")
    if unavailable:
        logger.warning(f"Codecs NOT available: {unavailable}")

    if not available:
        logger.error("No codecs available! Cannot run probe.")
        return 1

    # ---- Step 1: Load audio ----
    logger.info("\n--- Step 1: Loading audio ---")
    t0 = time.time()

    if args.synthetic:
        waveforms = generate_synthetic_audio(args.num_samples, args.seed)
    else:
        waveforms = load_real_audio(args.split, args.num_samples, args.seed)

    logger.info(f"Audio loaded in {time.time() - t0:.1f}s ({len(waveforms)} samples)")

    # ---- Step 2: Create augmented dataset ----
    logger.info("\n--- Step 2: Creating codec-augmented pairs ---")
    t0 = time.time()

    all_waveforms, binary_labels, multiclass_labels, codec_names = create_augmented_dataset(
        waveforms, supported_codecs, seed=args.seed,
    )

    logger.info(f"Augmentation completed in {time.time() - t0:.1f}s")

    # Log class distribution
    for name, labels in [("Binary", binary_labels), ("Multiclass", multiclass_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        logger.info(f"  {name} distribution: {dist}")

    # ---- Step 3: Extract features ----
    logger.info("\n--- Step 3: Extracting frozen backbone features ---")
    t0 = time.time()

    model, backbone_cfg = load_backbone(args.backbone)
    num_layers = backbone_cfg["num_layers"]

    layer_features = extract_all_layer_features(
        model, all_waveforms, batch_size=args.batch_size,
    )

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Feature extraction completed in {time.time() - t0:.1f}s")
    logger.info(f"  Feature shape per layer: {layer_features[0].shape}")

    # ---- Step 4: Train probes ----
    logger.info("\n--- Step 4: Training linear probes ---")
    t0 = time.time()

    results = {}
    for layer_idx in range(num_layers):
        X = layer_features[layer_idx]

        bin_result = train_linear_probe(X, binary_labels, seed=args.seed)
        mc_result = train_linear_probe(X, multiclass_labels, seed=args.seed)

        results[layer_idx] = {
            "binary": bin_result,
            "multiclass": mc_result,
        }

        bin_acc = bin_result.get("accuracy", float("nan"))
        mc_acc = mc_result.get("accuracy", float("nan"))
        logger.info(
            f"  Layer {layer_idx:2d}: binary={bin_acc:.4f}, multiclass={mc_acc:.4f}"
        )

    logger.info(f"Probing completed in {time.time() - t0:.1f}s")

    # ---- Step 5: Report ----
    best_binary, best_multi = print_results_table(results, num_layers, args.backbone)

    # ---- Step 6: Save results ----
    output = {
        "backbone": args.backbone,
        "pretrained": backbone_cfg["pretrained"],
        "num_layers": num_layers,
        "hidden_size": backbone_cfg["hidden_size"],
        "num_original_samples": len(waveforms),
        "num_total_samples": len(all_waveforms),
        "data_source": "synthetic" if args.synthetic else f"asvspoof5_{args.split}",
        "codecs_available": available,
        "codecs_unavailable": unavailable,
        "seed": args.seed,
        "binary_class_distribution": {
            int(k): int(v)
            for k, v in zip(*np.unique(binary_labels, return_counts=True))
        },
        "multiclass_class_distribution": {
            int(k): int(v)
            for k, v in zip(*np.unique(multiclass_labels, return_counts=True))
        },
        "per_layer": {
            str(k): v for k, v in results.items()
        },
        "best_binary_accuracy": float(best_binary),
        "best_multiclass_accuracy": float(best_multi),
        "interpretation": (
            "STRONG_SIGNAL" if best_binary > 0.70
            else "WEAK_SIGNAL" if best_binary > 0.60
            else "MARGINAL" if best_binary > 0.55
            else "NO_SIGNAL"
        ),
    }

    # Save to output dir
    results_path = args.output_dir / "domain_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Also save to project results/ directory if different
    project_root = Path(__file__).resolve().parent.parent
    canonical_path = project_root / "results" / "domain_probe_results.json"
    if canonical_path.resolve() != results_path.resolve():
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        with open(canonical_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        logger.info(f"Results also saved to: {canonical_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
