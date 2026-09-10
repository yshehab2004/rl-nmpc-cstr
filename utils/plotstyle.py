import matplotlib as mpl
import matplotlib.pyplot as plt

TIER_STYLE = {
    "T0a_naive":   {"color": "#2a78d6", "ls": "-",   "marker": "o", "label": "T0a naive fixed"},
    "T0b_prudent": {"color": "#eb6834", "ls": "--",  "marker": "s", "label": "T0b prudent fixed"},
    "T2_rto":      {"color": "#1baf7a", "ls": "-.",  "marker": "^", "label": "T2 RTO"},
    "T3_rl":       {"color": "#4a3aa7", "ls": ":",   "marker": "D", "label": "T3 RL"},
}

CONSTRAINT_COLOR = "#e34948"
REFERENCE_COLOR = "#52514e"
TRUTH_COLOR = "#0b0b0b"

BETA_CMAP = "Blues"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    })


def tier(name):
    return TIER_STYLE[name]


def limit_line(ax, y, label="T_R limit 140 degC", axis="h"):
    draw = ax.axhline if axis == "h" else ax.axvline
    draw(y, color=CONSTRAINT_COLOR, ls=(0, (4, 3)), lw=1.2, zorder=1.5, label=label)


def save(fig, path, params, script, start_time=None, extra=None):
    from utils.provenance import save_with_provenance

    fig.savefig(path)
    plt.close(fig)
    return save_with_provenance(path, data=None, params=params, script=script,
                                 start_time=start_time, extra=extra)
