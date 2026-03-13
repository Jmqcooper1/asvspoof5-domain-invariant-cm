#!/usr/bin/env python3
"""Visualize codec artifacts: waveform, spectrogram, spectral envelope, and residual.

Creates a 12-row × 4-column grid showing how each ASVspoof 5 codec condition
affects the same speaker's bonafide audio.

Usage:
    python scripts/plot_codec_artifacts.py \
        --data-root /projects/prjs1904/data/asvspoof5 \
        --predictions results/runs/wavlm_erm/eval_eval_full/predictions.tsv \
        --speaker E_0002 \
        --output figures/codec_artifacts
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.colorbar as mcolorbar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import COLORS, STYLE, set_style


# ── Codec metadata ───────────────────────────────────────────────────────────
CODEC_LABELS = {
    'NONE': 'No Codec',
    'C01':  'C01 · Opus 8 kHz',
    'C02':  'C02 · AMR-WB 8 kHz',
    'C03':  'C03 · Speex 8 kHz',
    'C04':  'C04 · Encodec',
    'C05':  'C05 · MP3 16 kHz',
    'C06':  'C06 · AAC 16 kHz',
    'C07':  'C07 · MP3 + Encodec',
    'C08':  'C08 · Opus NB',
    'C09':  'C09 · AMR-NB',
    'C10':  'C10 · Speex NB',
    'C11':  'C11 · Device Effects',
}

CODEC_ORDER = ['NONE', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07',
               'C08', 'C09', 'C10', 'C11']

# Colour per codec for spectral envelope overlay
CODEC_COLORS = {
    'NONE': '#333333',
    'C01':  '#D4795A', 'C02': '#E8946E', 'C03': '#F0A882',
    'C04':  '#7C3AED',
    'C05':  '#4CA08A', 'C06': '#6BC4AE',
    'C07':  '#9B59B6',
    'C08':  '#3B82F6', 'C09': '#60A5FA', 'C10': '#93C5FD',
    'C11':  '#F59E0B',
}


# ── Audio helpers ────────────────────────────────────────────────────────────
def load_audio(flac_path: Path, target_sr: int = 16000):
    try:
        import soundfile as sf
        audio, sr = sf.read(str(flac_path))
    except ImportError:
        import librosa
        audio, sr = librosa.load(str(flac_path), sr=target_sr)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return audio, sr


def compute_spectrogram(audio, sr, n_fft=1024, hop_length=256):
    window = np.hanning(n_fft)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))
    return 20 * np.log10(np.maximum(spec, 1e-10))


def compute_spectral_envelope(spec_db):
    """Mean power per frequency bin across all frames → spectral envelope."""
    return np.mean(spec_db, axis=1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Visualize codec artifacts')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--speaker', type=str, default='E_0002')
    parser.add_argument('--output', type=str, default='figures/codec_artifacts')
    parser.add_argument('--max-seconds', type=float, default=3.0)
    args = parser.parse_args()

    set_style()

    data_root = Path(args.data_root)
    eval_dir = data_root / 'flac_E_eval'
    if not eval_dir.exists():
        for candidate in ['flac_E', 'eval']:
            if (data_root / candidate).exists():
                eval_dir = data_root / candidate
                break

    import pandas as pd
    df = pd.read_csv(args.predictions, sep='\t')
    bon = df[(df['y_task'] == 0) & (df['speaker_id'] == args.speaker)]

    utterances = {}
    for codec in CODEC_ORDER:
        subset = bon[bon['codec'] == codec]
        if len(subset) == 0:
            print(f'Warning: no {codec} utterance for speaker {args.speaker}')
            continue
        utterances[codec] = subset.iloc[0]['flac_file']

    print(f'Speaker {args.speaker}: {len(utterances)}/{len(CODEC_ORDER)} codecs')

    if 'NONE' not in utterances:
        print('Error: no NONE reference'); sys.exit(1)

    ref_path = eval_dir / f'{utterances["NONE"]}.flac'
    ref_audio, sr = load_audio(ref_path)
    max_samples = int(args.max_seconds * sr)
    ref_audio = ref_audio[:max_samples]
    ref_spec = compute_spectrogram(ref_audio, sr)
    ref_envelope = compute_spectral_envelope(ref_spec)
    freq_bins = np.linspace(0, sr / 2, ref_spec.shape[0])
    print(f'Reference: {ref_path.name} ({len(ref_audio)/sr:.2f}s, {sr}Hz)')

    # ── Figure layout ────────────────────────────────────────────────────────
    n_codecs = len(utterances)
    row_h = 1.8
    fig_h = row_h * n_codecs + 1.5   # extra for suptitle + colorbar
    fig = plt.figure(figsize=(18, fig_h))

    # 4 columns: waveform | spectrogram | spectral envelope | residual
    gs = gridspec.GridSpec(
        n_codecs + 1, 4, figure=fig,
        width_ratios=[1, 1.3, 0.8, 1.3],
        height_ratios=[1] * n_codecs + [0.06],
        hspace=0.40, wspace=0.30,
    )

    vmin_spec, vmax_spec = -80, 0
    vmin_res, vmax_res = -30, 30

    for row_idx, codec in enumerate(CODEC_ORDER):
        if codec not in utterances:
            continue

        flac_path = eval_dir / f'{utterances[codec]}.flac'
        if not flac_path.exists():
            print(f'Warning: {flac_path} not found'); continue

        audio, _ = load_audio(flac_path)
        audio = audio[:max_samples]
        t = np.arange(len(audio)) / sr
        spec = compute_spectrogram(audio, sr)
        envelope = compute_spectral_envelope(spec)
        label = CODEC_LABELS.get(codec, codec)
        color = CODEC_COLORS.get(codec, '#888888')
        print(f'  {codec}: {flac_path.name} ({len(audio)/sr:.2f}s)')

        # ── Col 0: Waveform ──────────────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.plot(t, audio, color=color, linewidth=0.25, alpha=0.85)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, args.max_seconds)
        ax.set_ylabel(label, fontsize=9, fontweight='bold',
                      rotation=0, labelpad=115, ha='left', va='center')
        ax.yaxis.set_label_coords(-0.55, 0.5)
        if row_idx == 0:
            ax.set_title('Waveform', fontsize=12, fontweight='bold', pad=8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_yticks([-0.5, 0, 0.5])

        # ── Col 1: Spectrogram ───────────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 1])
        time_bins = np.linspace(0, len(audio) / sr, spec.shape[1])
        im_spec = ax.pcolormesh(
            time_bins, freq_bins / 1000, spec,
            cmap='magma', vmin=vmin_spec, vmax=vmax_spec,
            shading='gouraud', rasterized=True,
        )
        ax.set_ylim(0, 8)
        if row_idx == 0:
            ax.set_title('Spectrogram', fontsize=12, fontweight='bold', pad=8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('kHz', fontsize=9)
        ax.tick_params(labelsize=8)

        # ── Col 2: Spectral envelope ─────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 2])
        # Plot reference (NONE) as thin grey, current codec as coloured
        ax.plot(ref_envelope, freq_bins / 1000, color='#CCCCCC',
                linewidth=1.2, alpha=0.7, label='No Codec' if row_idx == 0 else None)
        ax.plot(envelope, freq_bins / 1000, color=color,
                linewidth=1.8, alpha=0.9)
        ax.set_ylim(0, 8)
        ax.set_xlim(-80, 0)
        if row_idx == 0:
            ax.set_title('Spectral Envelope', fontsize=12, fontweight='bold', pad=8)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Power (dB)', fontsize=10)
        ax.set_ylabel('kHz', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.axhline(y=4, color=STYLE['GRID'], linewidth=0.5, linestyle='--', alpha=0.5)

        # ── Col 3: Spectral residual ─────────────────────────────────────
        ax = fig.add_subplot(gs[row_idx, 3])
        if codec == 'NONE':
            ax.set_facecolor(STYLE['PLOT_BG'])
            ax.text(0.5, 0.5, '(reference)', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#9CA3AF',
                    fontstyle='italic')
            ax.set_ylim(0, 8)
            ax.set_xlim(0, args.max_seconds)
        else:
            min_frames = min(spec.shape[1], ref_spec.shape[1])
            residual = spec[:, :min_frames] - ref_spec[:, :min_frames]
            time_res = np.linspace(0, min(len(audio), len(ref_audio)) / sr, min_frames)
            im_res = ax.pcolormesh(
                time_res, freq_bins / 1000, residual,
                cmap='RdBu_r', vmin=vmin_res, vmax=vmax_res,
                shading='gouraud', rasterized=True,
            )
            ax.set_ylim(0, 8)

        if row_idx == 0:
            ax.set_title('Spectral Residual', fontsize=12, fontweight='bold', pad=8)
        if row_idx < n_codecs - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('kHz', fontsize=9)
        ax.tick_params(labelsize=8)

    # ── Colorbars (bottom row) ───────────────────────────────────────────────
    ax_cb_spec = fig.add_subplot(gs[n_codecs, 1])
    cb1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin_spec, vmax_spec), cmap='magma'),
        cax=ax_cb_spec, orientation='horizontal',
    )
    cb1.set_label('Power (dB)', fontsize=9)
    cb1.ax.tick_params(labelsize=8)

    ax_cb_res = fig.add_subplot(gs[n_codecs, 3])
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(norm=Normalize(vmin_res, vmax_res), cmap='RdBu_r'),
        cax=ax_cb_res, orientation='horizontal',
    )
    cb2.set_label('Δ Power (dB)', fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    # Hide unused bottom cells
    for col in [0, 2]:
        ax_empty = fig.add_subplot(gs[n_codecs, col])
        ax_empty.axis('off')

    fig.suptitle(
        f'Codec Artifacts Across ASVspoof 5 Conditions — Speaker {args.speaker} (Bonafide)',
        fontsize=14, fontweight='bold', y=1.005,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    output_base = args.output.removesuffix('.png').removesuffix('.pdf')
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    for ext in ['.png', '.pdf']:
        out = output_base + ext
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved: {out}')
    plt.close()


if __name__ == '__main__':
    main()
