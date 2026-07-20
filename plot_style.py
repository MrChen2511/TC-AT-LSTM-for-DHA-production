import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def set_style():
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['savefig.dpi'] = 600
    matplotlib.rcParams['savefig.format'] = 'pdf'
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['savefig.pad_inches'] = 0.05

    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['lines.markersize'] = 6

    matplotlib.rcParams['axes.linewidth'] = 1.0
    matplotlib.rcParams['axes.labelsize'] = 11
    matplotlib.rcParams['axes.titlesize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 11
    matplotlib.rcParams['ytick.labelsize'] = 11
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['xtick.major.width'] = 0.8
    matplotlib.rcParams['ytick.major.width'] = 0.8

    matplotlib.rcParams['legend.fontsize'] = 10
    matplotlib.rcParams['legend.frameon'] = False

    matplotlib.rcParams['axes.grid'] = False

    print('[plot_style] rcParams configured (DejaVu Sans, DPI 600, PDF)')


MODEL_CATEGORY_COLORS = {
    'mechanistic': '#E69F00',
    'ml':          '#56B4E9',
    'dl':          '#009E73',
    'hybrid':      '#D55E00',
}

MODEL_COLORS_10 = {
    'M-LP':   '#E69F00', 'M-RF':   '#56B4E9', 'M-XGB':  '#0072B2',
    'M-LGB':  '#56B4E9', 'M-SVR':  '#009E73', 'M-RNN':  '#009E73',
    'M-LSTM': '#F0E442', 'M-CNNL': '#009E73', 'M-ATL':  '#CC79A7',
    'M-TCAL': '#D55E00',
}

FEATURE_COLORS_8 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]

QUALITATIVE_10 = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',
]


def get_color_palette(name):
    palettes = {
        'model_category': MODEL_CATEGORY_COLORS,
        'model_10':       MODEL_COLORS_10,
        'feature_8':      FEATURE_COLORS_8,
        'qualitative_10': QUALITATIVE_10,
    }
    if name not in palettes:
        raise KeyError(f"Unknown palette: {name}. Options: {list(palettes.keys())}")
    return palettes[name]


def save_figure(fig, path, dpi=600, fmt='pdf', close=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format=fmt)
    print(f'Saved: {path}')
    if close:
        plt.close(fig)


FIG_SIZES = {
    'single':    (6.5, 4.0), 'double': (6.5, 8.0), 'wide': (7.2, 3.5),
    'full_page': (7.2, 9.5), 'square': (5.0, 5.0), 'heatmap': (6.5, 5.5),
    'half_page': (7.2, 4.5),
}


def get_figsize(name):
    if name not in FIG_SIZES:
        raise KeyError(f"Unknown size: {name}. Options: {list(FIG_SIZES.keys())}")
    return FIG_SIZES[name]