"""Static, exportable audit figures; labels identify demonstration output."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def audit_scatter(frame, x, y, output, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for name, subset in frame.groupby("protocol"):
        ax.scatter(subset[x], subset[y], label=name, alpha=0.65, s=26)
    low = min(frame[x].min(), frame[y].min())
    high = max(frame[x].max(), frame[y].max())
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    fig.savefig(str(output).replace(".png", ".svg"))
    plt.close(fig)
