"""Shared publication style for journal figures (BMC: 85 mm single / 170 mm double column, sans-serif, 600 dpi).
Colours are fixed per entity and were checked with the dataviz palette validator (light mode, all pairs):
designs / cohort pairs: #2a78d6, #eb6834, #1baf7a (all checks pass); outcome mechanisms: #6250d6, #1baf7a, #e87ba4
(CVD 6.1 -> always paired with distinct markers and line styles). Every series also has a legend or direct label."""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 85 * MM, 170 * MM
INK, INK2, MUTED, GRID = "#1a1a1a", "#52514e", "#8c8b86", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"

DESIGN_C = {"random_compound": BLUE, "scaffold": ORANGE, "chemical_cluster": AQUA}
DESIGN_L = {"random_compound": "Random-compound", "scaffold": "Scaffold", "chemical_cluster": "Chemical-cluster"}
DESIGN_M = {"random_compound": "o", "scaffold": "s", "chemical_cluster": "^"}
SPLIT_C = {"held_out_donor": BLUE, "random_row": ORANGE}
SPLIT_L = {"held_out_donor": "Held-out-donor", "random_row": "Random-row"}
SPLIT_M = {"held_out_donor": "o", "random_row": "s"}
MECH_C = {"linear": VIOLET, "interaction": AQUA, "ptm_dominant": MAGENTA}
MECH_L = {"linear": "Linear", "interaction": "Interaction", "ptm_dominant": "PTM-dominant"}
MECH_M = {"linear": "o", "interaction": "s", "ptm_dominant": "D"}
MECH_LS = {"linear": "-", "interaction": "--", "ptm_dominant": ":"}
MODEL_L = {"molecular_only": "Molecular-only", "context_only": "Context-only", "early_fusion": "Early fusion", "without_ptm": "No-PTM", "late_fusion": "Late fusion",
           "unweighted_logistic": "Unweighted logistic", "confidence_weighted_logistic": "Confidence-weighted logistic", "expression_only": "Expression-only logistic",
           "random_forest": "Random forest"}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.2, "lines.markersize": 4, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 4, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    "errorbar.capsize": 0,
})


def panel(ax, letter, title=None, x=-0.16, y=1.06):
    """Bold upper-case panel letter at the top left, optional short panel title beside it."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)
    if title:
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=4)


def legend_below(ax, handles=None, ncol=1, dy=-0.2, **kw):
    """Legend placed under the axes (below the x-axis label) so it never covers data."""
    args = dict(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, columnspacing=1.0, handletextpad=0.4)
    args.update(kw)
    return ax.legend(handles=handles, **args) if handles is not None else ax.legend(**args)


def design_handles(order, colors, labels, markers):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=colors[k], marker=markers[k], lw=0, ms=4, label=labels[k]) for k in order]


def refline(ax, y=None, x=None, label=None, **kw):
    style = dict(color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    style.update(kw)
    if y is not None:
        ax.axhline(y, **style)
        if label:
            ax.text(ax.get_xlim()[1], y, " " + label, va="center", ha="left", fontsize=6, color=INK2, clip_on=False)
    if x is not None:
        ax.axvline(x, **style)


def save(fig, out_stem):
    """Write PNG (600 dpi), TIFF (LZW, from the PNG) and vector PDF; return the PNG path."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08, facecolor="white")
    Image.open(png).convert("RGB").save(out_stem.with_suffix(".tif"), compression="tiff_lzw", dpi=(600, 600))
    plt.close(fig)
    return png


def box(ax, xy, w, h, text, fc="#f4f3f0", ec=INK2, fs=6.5, bold_first=True, lw=0.7, ha="center"):
    """Rounded schematic box; first line bold."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=lw, transform=ax.transAxes, zorder=2))
    lines = text.split("\n")
    cx = xy[0] + (w / 2 if ha == "center" else 0.012)
    top = xy[1] + h - 0.022
    step = min(0.052, (h - 0.03) / max(len(lines), 1))
    for i, ln in enumerate(lines):
        ax.text(cx, top - i * step, ln, transform=ax.transAxes, ha=ha, va="top", fontsize=fs + (0.5 if (i == 0 and bold_first) else 0),
                fontweight="bold" if (i == 0 and bold_first) else "normal", color=INK, zorder=3)


def arrow(ax, a, b, color=INK2, lw=0.8, style="-|>", rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(a, b, transform=ax.transAxes, arrowstyle=style, mutation_scale=7, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))
