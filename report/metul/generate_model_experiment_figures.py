"""Generate figures for report/metul/model_experiments.md.

All plotted values are copied from executed notebook outputs. This script
does not fit, load, or score any model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
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


# Figure 1: representative listing-level models on the May-2024-forward split.
labels = [
    "Exp. 7 HPO broad\nAutoGluon",
    "Exp. 9 default-setting\nAutoGluon",
    "Exp. 14 weighted\nresidual stack",
    "Exp. 10 equal\nGBM/CAT blend",
    "Exp. 12 equal\nDL blend",
    "Exp. 10 neural\nGBM/CAT gate",
    "Exp. 11 town-\nrouted AutoGluon",
    "Exp. 13 tuned\nrandom forest",
    "Exp. 10\nTabTransformer",
    "Exp. 14 AG +\nAdaBoost residual",
]
values = [466.432, 467.122, 468.057, 468.740, 471.168, 472.274, 475.846, 479.319, 495.853, 497.420]
colors = [GREEN, ORANGE, ORANGE, BLUE, BLUE, BLUE, BLUE, GREY, RED, RED]

figure, axis = plt.subplots(figsize=(10.8, 7.2))
positions = np.arange(len(labels))
bars = axis.barh(positions, values, color=colors, alpha=0.92)
axis.set_yticks(positions, labels)
axis.invert_yaxis()
axis.set_xlim(460, 502)
axis.set_xlabel("Validation RMSE")
axis.set_title("Representative listing-level model results on the common chronological split")
axis.tick_params(axis="y", labelsize=9)
for bar, value in zip(bars, values):
    axis.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", ha="left", va="center", fontsize=8.5)
axis.text(
    0.01,
    -0.13,
    "The split is common, but training budgets and feature-generator settings changed across experiments; controlled conclusions use within-experiment comparisons.",
    transform=axis.transAxes,
    fontsize=9,
    color="#475467",
)
save(figure, "model_family_rmse.png")


# Figure 2: approaches built from the same 468.272 AutoGluon base prediction.
labels = [
    "Base\nAutoGluon",
    "Weighted global\nresidual",
    "Weighted flat-type\nresidual experts",
    "Significant-residual\ngate",
    "Unweighted global\nresidual",
    "AdaBoost\nresidual",
    "Prediction-as-feature\nsecond layer",
]
values = [468.272, 468.057, 468.068, 468.360, 473.949, 497.420, 538.151]
colors = [GREY, GREEN, GREEN, BLUE, RED, RED, RED]

figure, axis = plt.subplots(figsize=(11.8, 6.2))
positions = np.arange(len(labels))
bars = axis.bar(positions, values, color=colors, alpha=0.92)
axis.axhline(values[0], color="#344054", linewidth=1.2, linestyle="--", label="Layer-1 reference")
axis.set_xticks(positions, labels)
axis.set_ylim(460, 545)
axis.set_ylabel("Validation RMSE")
axis.set_title("Residual and second-layer strategies using the same Layer-1 model")
axis.tick_params(axis="x", labelsize=8.5)
axis.legend(frameon=False, loc="upper left")
for bar, value in zip(bars, values):
    axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
save(figure, "residual_model_rmse.png")

print(f"Generated model-experiment figures in {FIGURE_DIR}")
