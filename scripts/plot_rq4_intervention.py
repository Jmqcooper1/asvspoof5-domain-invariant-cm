#!/usr/bin/env python3
"""Generate RQ4 visualizations: CKA analysis and activation patching results.

This script creates publication-ready figures for RQ4:
"Where does DANN achieve domain invariance?"

Figures generated:
1. CKA heatmap: Layer-by-layer similarity between ERM and DANN
2. Intervention ablation: Effect of patching on EER and probe accuracy
3. Layer contribution divergence: Bar chart showing per-layer CKA

Key findings visualized:
- Layer 11 shows dramatic divergence (CKA=0.098) despite frozen backbone
- Projection layer patching reduces domain leakage while preserving EER
- DANN's effect is concentrated in pooling weights + projection head

Usage:
    python scripts/plot_rq4_intervention.py \\
        --cka-results rq4_cka_results.csv \\
        --intervention-results rq4_results_summary.csv \\
        --output-dir figures/rq4

    # From Snellius results
    python scripts/plot_rq4_intervention.py \\
        --cka-results /projects/prjs1904/runs/wavlm_dann/rq4_cka_results.csv \\
        --intervention-results /projects/prjs1904/runs/wavlm_dann/rq4_results_summary.csv \\
        --output-dir figures/rq4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Publication-quality settings
FIGSIZE_SINGLE = (6, 4)
FIGSIZE_WIDE = (10, 4)
FIGSIZE_TALL = (6, 6)
DPI = 300

# Color palette (consistent with other thesis figures)
COLORS = {
    'erm': '#2ecc71',      # Green
    'dann': '#3498db',     # Blue
    'divergent': '#e74c3c', # Red for divergent layers
    'neutral': '#95a5a6',   # Gray
}


def plot_cka_layer_bar(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart showing per-layer CKA between ERM and DANN.
    
    Highlights layer 11's dramatic divergence.
    """
    # Filter to pool_weight_transplant layer_contrib (the meaningful comparison)
    layer_data = df[
        (df['mode'] == 'pool_weight_transplant') & 
        (df['representation_mode'] == 'layer_contrib')
    ].copy()
    
    if layer_data.empty:
        logger.warning("No pool_weight_transplant layer_contrib data found")
        return
    
    # Sort by layer key numerically
    layer_data['layer_num'] = layer_data['layer_key'].astype(int)
    layer_data = layer_data.sort_values('layer_num')
    
    layers = layer_data['layer_num'].values
    cka_values = layer_data['cka'].values
    
    # Color bars based on divergence
    colors = [COLORS['divergent'] if cka < 0.5 else COLORS['dann'] for cka in cka_values]
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    bars = ax.bar(layers, cka_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add horizontal reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Identical')
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5, label='Moderate similarity')
    
    # Annotate layer 11
    layer_11_idx = np.where(layers == 11)[0]
    if len(layer_11_idx) > 0:
        idx = layer_11_idx[0]
        ax.annotate(
            f'CKA={cka_values[idx]:.2f}',
            xy=(11, cka_values[idx]),
            xytext=(9, 0.3),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10,
            fontweight='bold',
            color='red'
        )
    
    ax.set_xlabel('Transformer Layer', fontsize=12)
    ax.set_ylabel('CKA Similarity (ERM vs DANN)', fontsize=12)
    ax.set_title('Layer Contribution Similarity: ERM vs DANN', fontsize=14, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left')
    
    # Add interpretation text
    ax.text(
        0.98, 0.02,
        'Layer 11: DANN diverges\n(codec info removed)',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=9,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CKA layer bar chart: {output_path}")


def plot_intervention_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped bar chart comparing intervention effects on EER and probe accuracy."""
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    # Prepare data
    modes = df['mode'].values
    mode_labels = {
        'layer_patch_hidden': 'Baseline\n(no patch)',
        'layer_patch_repr': 'Patch\nProjection',
        'layer_patch_mixed': 'Patch\nMixed',
        'pool_weight_transplant': 'Transplant\nPool Weights'
    }
    labels = [mode_labels.get(m, m) for m in modes]
    
    eer_values = df['eer'].values * 100  # Convert to percentage
    probe_values = df['max_probe_acc'].values * 100
    
    x = np.arange(len(modes))
    width = 0.6
    
    # Left panel: EER
    colors_eer = [COLORS['neutral'] if i == 0 else COLORS['dann'] for i in range(len(modes))]
    axes[0].bar(x, eer_values, width, color=colors_eer, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('EER (%)', fontsize=12)
    axes[0].set_title('Spoofing Detection Performance', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, max(eer_values) * 1.2)
    
    # Add value labels
    for i, v in enumerate(eer_values):
        axes[0].text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # Right panel: Probe accuracy (domain leakage)
    colors_probe = [COLORS['divergent'] if v > 50 else COLORS['erm'] for v in probe_values]
    axes[1].bar(x, probe_values, width, color=colors_probe, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Codec Probe Accuracy (%)', fontsize=12)
    axes[1].set_title('Domain Information Leakage', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(0, 100)
    
    # Add chance level line
    axes[1].axhline(y=33.3, color='gray', linestyle='--', alpha=0.7, label='Chance (3 codecs)')
    axes[1].legend(loc='upper right')
    
    # Add value labels
    for i, v in enumerate(probe_values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Highlight the key finding
    repr_idx = list(modes).index('layer_patch_repr') if 'layer_patch_repr' in modes else None
    if repr_idx is not None:
        axes[1].annotate(
            'Domain invariance\nachieved here',
            xy=(repr_idx, probe_values[repr_idx]),
            xytext=(repr_idx + 0.8, probe_values[repr_idx] + 15),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=9,
            color='green',
            fontweight='bold'
        )
    
    plt.suptitle('RQ4: Activation Patching Ablation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved intervention comparison: {output_path}")


def plot_cka_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Create heatmap showing CKA across different intervention modes."""
    
    # Get unique modes and their CKA summaries
    modes = df['mode'].unique()
    
    # For each mode, get the mean/summary CKA
    summary_data = []
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        if 'layer_key' in mode_df.columns:
            # Has layer-wise data
            for _, row in mode_df.iterrows():
                summary_data.append({
                    'mode': mode,
                    'layer': str(row['layer_key']),
                    'cka': row['cka']
                })
        else:
            summary_data.append({
                'mode': mode,
                'layer': 'all',
                'cka': mode_df['cka'].mean()
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Pivot for heatmap (only for pool_weight_transplant which has per-layer data)
    layer_data = summary_df[summary_df['mode'] == 'pool_weight_transplant'].copy()
    
    if layer_data.empty:
        logger.warning("No layer-wise data for heatmap")
        return
    
    layer_data['layer_num'] = pd.to_numeric(layer_data['layer'], errors='coerce')
    layer_data = layer_data.dropna(subset=['layer_num'])
    layer_data = layer_data.sort_values('layer_num')
    
    # Create simple heatmap (1 row x 12 layers)
    fig, ax = plt.subplots(figsize=(10, 2))
    
    cka_matrix = layer_data['cka'].values.reshape(1, -1)
    
    im = ax.imshow(cka_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'L{i}' for i in range(12)])
    ax.set_yticks([0])
    ax.set_yticklabels(['ERM↔DANN'])
    ax.set_xlabel('Transformer Layer')
    ax.set_title('CKA Similarity: Layer Contributions (ERM vs DANN)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.3, shrink=0.8)
    cbar.set_label('CKA Similarity')
    
    # Annotate low CKA values
    for i, cka in enumerate(layer_data['cka'].values):
        color = 'white' if cka < 0.5 else 'black'
        ax.text(i, 0, f'{cka:.2f}', ha='center', va='center', fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved CKA heatmap: {output_path}")


def plot_delta_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot: Δ EER vs Δ Probe accuracy for each intervention."""
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    # Filter out baseline (delta = 0)
    plot_df = df[df['mode'] != 'layer_patch_hidden'].copy()
    
    delta_eer = plot_df['delta_eer_vs_base'].values * 100
    delta_probe = plot_df['delta_probe_vs_base'].values * 100
    modes = plot_df['mode'].values
    
    mode_labels = {
        'layer_patch_repr': 'Projection\nPatch',
        'layer_patch_mixed': 'Mixed\nPatch',
        'pool_weight_transplant': 'Pool Weight\nTransplant'
    }
    
    colors = [COLORS['erm'], COLORS['dann'], COLORS['divergent']]
    
    for i, (de, dp, mode) in enumerate(zip(delta_eer, delta_probe, modes)):
        ax.scatter(de, dp, s=200, c=colors[i % len(colors)], edgecolor='black', linewidth=1.5, zorder=3)
        ax.annotate(
            mode_labels.get(mode, mode),
            xy=(de, dp),
            xytext=(de + 0.1, dp + 2),
            fontsize=9,
            ha='left'
        )
    
    # Add quadrant lines
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Quadrant labels
    ax.text(0.02, 0.98, 'Better EER\nMore leakage', transform=ax.transAxes, 
            ha='left', va='top', fontsize=8, color='gray', style='italic')
    ax.text(0.98, 0.98, 'Worse EER\nMore leakage', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='gray', style='italic')
    ax.text(0.02, 0.02, 'Better EER\nLess leakage ✓', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=8, color='green', style='italic', fontweight='bold')
    ax.text(0.98, 0.02, 'Worse EER\nLess leakage', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='gray', style='italic')
    
    ax.set_xlabel('Δ EER (%) vs Baseline', fontsize=12)
    ax.set_ylabel('Δ Probe Accuracy (%) vs Baseline', fontsize=12)
    ax.set_title('Intervention Trade-offs: Detection vs Domain Invariance', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    logger.info(f"Saved delta scatter: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cka-results', type=Path, required=True, help='Path to rq4_cka_results.csv')
    parser.add_argument('--intervention-results', type=Path, required=True, help='Path to rq4_results_summary.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('figures/rq4'), help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading CKA results from {args.cka_results}")
    cka_df = pd.read_csv(args.cka_results)
    
    logger.info(f"Loading intervention results from {args.intervention_results}")
    intervention_df = pd.read_csv(args.intervention_results)
    
    # Generate figures
    plot_cka_layer_bar(cka_df, args.output_dir / 'cka_layer_divergence.png')
    plot_cka_heatmap(cka_df, args.output_dir / 'cka_heatmap.png')
    plot_intervention_comparison(intervention_df, args.output_dir / 'intervention_comparison.png')
    plot_delta_scatter(intervention_df, args.output_dir / 'intervention_tradeoff.png')
    
    logger.info(f"All RQ4 figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
