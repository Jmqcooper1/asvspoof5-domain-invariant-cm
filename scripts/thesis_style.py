"""
Thesis figure style — Anthropic-inspired academic palette.
Usage: from thesis_style import STYLE, set_style, COLORS
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    'WAVLM_ERM':  '#D4795A',  # Terracotta
    'WAVLM_DANN': '#4CA08A',  # Muted teal
    'W2V2_ERM':   '#E8B4A0',  # Light terracotta
    'W2V2_DANN':  '#A5D5C3',  # Light teal
    'BASELINE':   '#9CA3AF',  # Gray
    'ACCENT':     '#3B4D6B',  # Dark navy
    'HIGHLIGHT':  '#E74C3C',  # Red for outlier highlight
}

STYLE = {
    'BG':      '#FAFAFA',
    'PLOT_BG': '#F5F5F5',
    'GRID':    '#E0E0E0',
    'AXIS':    '#444444',
    'TEXT':    '#333333',
    'TICK':    '#555555',
}

# CKA sequential: light blue → dark blue
CKA_CMAP_COLORS = ['#F0F4FF', '#C7D7FE', '#93B4FD', '#5B8DEF', '#2563EB', '#1D4ED8', '#1E3A8A']


def set_style():
    """Apply thesis matplotlib style globally."""
    mpl.rcParams.update({
        'figure.facecolor': STYLE['BG'],
        'axes.facecolor': STYLE['PLOT_BG'],
        'axes.edgecolor': STYLE['AXIS'],
        'axes.labelcolor': STYLE['AXIS'],
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'axes.axisbelow': True,
        'grid.color': STYLE['GRID'],
        'grid.linewidth': 0.5,
        'grid.alpha': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': STYLE['TICK'],
        'ytick.color': STYLE['TICK'],
        'text.color': STYLE['TEXT'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'savefig.facecolor': STYLE['BG'],
    })


# Hard-coded data extracted from existing figures and tables
DATA = {
    'main_results': {
        'WavLM ERM':    {'dev': 3.26, 'eval': 8.47, 'mindcf': 0.6388},
        'WavLM DANN':   {'dev': 4.76, 'eval': 7.34, 'mindcf': 0.5853},
        'W2V2 ERM':     {'dev': 4.24, 'eval': 15.30, 'mindcf': 1.0000},
        'W2V2 DANN':    {'dev': 4.45, 'eval': 14.33, 'mindcf': 1.0000},
        'WavLM ERM+Aug':{'dev': 3.26, 'eval': 7.98, 'mindcf': 0.6052},
        'W2V2 ERM+Aug': {'dev': 4.34, 'eval': 18.02, 'mindcf': 0.9992},
    },
    'cka': {
        0: 0.93, 1: 0.98, 2: 0.97, 3: 0.98, 4: 0.97, 5: 0.96,
        6: 0.90, 7: 0.82, 8: 0.83, 9: 0.81, 10: 0.73, 11: 0.06,
    },
    'pooling_weights': {
        'ERM':  [12.7, 11.4, 11.1, 10.6, 9.4, 8.5, 7.4, 6.8, 6.4, 5.8, 5.3, 4.6],
        'DANN': [7.6, 9.7, 8.1, 10.9, 11.0, 9.9, 8.6, 8.0, 8.0, 7.2, 5.8, 5.1],
    },
    'projection_probes': {
        'WavLM ERM': 0.434, 'WavLM DANN': 0.388,
        'W2V2 ERM': 0.471, 'W2V2 DANN': 0.460,
    },
    'projection_probes_std': {
        'WavLM ERM': 0.009, 'WavLM DANN': 0.009,
        'W2V2 ERM': 0.010, 'W2V2 DANN': 0.010,
    },
    'per_codec': {
        'C01': {'WavLM ERM': 7.49, 'WavLM DANN': 6.63, 'W2V2 ERM': 12.15, 'W2V2 DANN': 11.51},
        'C02': {'WavLM ERM': 5.88, 'WavLM DANN': 5.27, 'W2V2 ERM': 12.00, 'W2V2 DANN': 10.26},
        'C03': {'WavLM ERM': 7.90, 'WavLM DANN': 6.88, 'W2V2 ERM': 13.50, 'W2V2 DANN': 12.22},
        'C04': {'WavLM ERM': 10.68, 'WavLM DANN': 9.32, 'W2V2 ERM': 14.82, 'W2V2 DANN': 13.46},
        'C05': {'WavLM ERM': 6.09, 'WavLM DANN': 4.54, 'W2V2 ERM': 10.76, 'W2V2 DANN': 10.08},
        'C06': {'WavLM ERM': 6.85, 'WavLM DANN': 5.43, 'W2V2 ERM': 12.91, 'W2V2 DANN': 12.80},
        'C07': {'WavLM ERM': 12.50, 'WavLM DANN': 11.16, 'W2V2 ERM': 16.44, 'W2V2 DANN': 15.04},
        'C08': {'WavLM ERM': 10.22, 'WavLM DANN': 9.66, 'W2V2 ERM': 22.60, 'W2V2 DANN': 13.65},
        'C09': {'WavLM ERM': 9.97, 'WavLM DANN': 9.10, 'W2V2 ERM': 19.02, 'W2V2 DANN': 14.20},
        'C10': {'WavLM ERM': 10.90, 'WavLM DANN': 10.37, 'W2V2 ERM': 21.58, 'W2V2 DANN': 15.18},
        'C11': {'WavLM ERM': 8.19, 'WavLM DANN': 6.85, 'W2V2 ERM': 22.03, 'W2V2 DANN': 13.04},
        'NONE': {'WavLM ERM': 5.68, 'WavLM DANN': 4.19, 'W2V2 ERM': 9.95, 'W2V2 DANN': 9.94},
    },
}
