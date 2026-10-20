"""Generate figures for report/metul/data_processing.md.

Every value plotted below is copied from an executed notebook output or from
the saved AutoGluon predictor produced by the corresponding experiment. No
model is fitted by this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#3568A8"
ORANGE = "#E27A3F"
GREEN = "#3C8D73"
RED = "#C8514B"
GREY = "#667085"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": ":",
    }
)


def save(figure: plt.Figure, filename: str) -> None:
    figure.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# Figure 1: the implemented preprocessing and feature-selection flow.
figure, axis = plt.subplots(figsize=(12.5, 7.2))
axis.set_xlim(0, 12.5)
axis.set_ylim(0, 7.2)
axis.axis("off")

nodes = [
    (0.4, 5.55, 2.25, 0.9, "Rental train/test\n+ auxiliary sources", GREY),
    (3.15, 5.55, 2.25, 0.9, "Normalize categorical\nstrings", BLUE),
    (5.9, 5.55, 2.25, 0.9, "Derive date, lease and\nflat-type variables", BLUE),
    (8.65, 5.55, 2.95, 0.9, "Join HDB attributes with\nfit-only fallback statistics", BLUE),
    (8.65, 3.75, 2.95, 0.9, "Add spatial distances/counts\nand monthly market data", GREEN),
    (5.9, 3.75, 2.25, 0.9, "Chronological feature-\nsubset selection", GREEN),
    (3.15, 3.75, 2.25, 0.9, "13-feature compact\nrepresentation", ORANGE),
    (3.15, 1.65, 2.25, 0.9, "Fit-only target history\n(mean and 24-month lag)", RED),
    (5.9, 1.65, 2.25, 0.9, "Nearest MRT, attraction\nand monument names", RED),
    (8.65, 1.65, 2.95, 0.9, "Active 18-feature input\nfor downstream models", ORANGE),
]

for x, y, width, height, label, color in nodes:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.93,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", color="white", fontsize=9.5)

arrows = [
    ((2.65, 6.0), (3.15, 6.0)),
    ((5.40, 6.0), (5.90, 6.0)),
    ((8.15, 6.0), (8.65, 6.0)),
    ((10.13, 5.55), (10.13, 4.65)),
    ((8.65, 4.20), (8.15, 4.20)),
    ((5.90, 4.20), (5.40, 4.20)),
    ((4.28, 3.75), (4.28, 2.55)),
    ((5.40, 2.10), (5.90, 2.10)),
    ((8.15, 2.10), (8.65, 2.10)),
]
for start, end in arrows:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#344054"))

axis.text(6.25, 6.92, "Implemented preprocessing and feature-selection pipeline", ha="center", va="center", fontsize=13, weight="bold")
axis.text(6.25, 0.72, "All fitted statistics are learned from the training partition; validation and test are transform-only.", ha="center", va="center", fontsize=9.5, color="#475467")
save(figure, "data_processing_pipeline.png")


# Figure 2: controlled comparisons within each experiment family.
panels = [
    (
        "Exp. 7: feature subsets",
        ["compact\n13", "compact +\ncoordinates 15", "full\n31", "expanded\n25"],
        [467.230, 467.480, 467.507, 468.304],
        [ORANGE, BLUE, BLUE, BLUE],
        (466.5, 469.0),
    ),
    (
        "Exp. 9: target-history variants",
        ["global lag\n+ mean + names", "compact\nreference", "town/flat lag\n+ mean + names"],
        [467.122, 467.230, 468.986],
        [ORANGE, GREY, BLUE],
        (466.5, 469.6),
    ),
    (
        "Exp. 14: external/encoding variants",
        ["native\ncategories", "+ resale\nfeatures", "normal target\nencoding", "historical target\nencoding"],
        [467.122, 467.563, 473.247, 477.800],
        [ORANGE, BLUE, RED, RED],
        (466.0, 479.0),
    ),
    (
        "Exp. 16: SHAP subset search",
        ["SHAP\ntop 18", "manual Exp. 9\n18", "SHAP\ntop 22", "all expanded\n36"],
        [467.841, 468.272, 468.721, 468.921],
        [ORANGE, GREY, BLUE, BLUE],
        (467.2, 469.4),
    ),
]

figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.7), constrained_layout=True)
for axis, (title, labels, values, colors, limits) in zip(axes.ravel(), panels):
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=colors, alpha=0.90)
    axis.set_xticks(positions, labels)
    axis.set_ylim(*limits)
    axis.set_ylabel("Validation RMSE")
    axis.set_title(title)
    axis.tick_params(axis="x", labelsize=8.5)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + (limits[1] - limits[0]) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
save(figure, "feature_processing_rmse.png")

print(f"Generated preprocessing figures in {FIGURE_DIR}")
