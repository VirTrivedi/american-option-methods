"""One matplotlib style and palette for every figure in the repo (imported by each script that draws)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"blue": "#4C78A8", "orange": "#F58518", "green": "#54A24B", "purple": "#B279A2", "grey": "#BBBBBB", "dark": "#444444"}
EXPLORATORY = "exploratory (post-hoc)"
DPI = 150

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": DPI, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "image.cmap": "viridis",
})


def footer(fig: plt.Figure, text: str) -> None:
    """Provenance line at the bottom of the figure."""
    fig.text(0.01, -0.01, text, fontsize=6.5, color="#555555", ha="left", va="top")


MANIFEST = "figure_manifest.csv"


def record(path: Path, provenance: str) -> None:
    """Upsert (figure name, provenance text) into figure_manifest.csv next to the figure. README captions quote this text, so N, snapshot date and source cannot drift from the plot."""
    import csv
    manifest = Path(path).parent / MANIFEST
    rows = {}
    if manifest.exists():
        with manifest.open(newline="") as f:
            rows = {r["figure"]: r["provenance"] for r in csv.DictReader(f)}

    rows[Path(path).stem] = provenance
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "provenance"])
        w.writerows(sorted(rows.items()))


def save(fig: plt.Figure, path: Path, provenance: str | None = None) -> None:
    if provenance:
        footer(fig, provenance)
        record(path, provenance)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)