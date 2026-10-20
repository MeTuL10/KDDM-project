"""Generate notebook-backed figures used by report/metul/eda.md.

The figures reproduce tables saved in the executed Experiment 13 and
Experiment 15 notebooks. Keeping those values explicit makes the report
figures reproducible without retraining the models that produced them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#3568A8"
ORANGE = "#E27A3F"
RED = "#C8514B"

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
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
    }
)


def save(figure: plt.Figure, filename: str) -> None:
    figure.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(figure)


# Figure 1: fit/validation association stability from Experiment 13.
stability = pd.DataFrame(
    {
        "feature": [
            "town × flat-type historical mean",
            "approval date number",
            "COE premium mean",
            "24-month global target lag",
            "floor area",
            "year completed",
            "maximum floor",
            "nearest mall distance",
            "nearest MRT distance",
            "approval month",
        ],
        "fit": [0.663695, 0.562226, 0.528425, 0.317593, 0.308192, 0.228823, 0.227813, -0.090170, -0.085130, 0.055160],
        "validation": [0.668246, 0.017877, 0.005706, 0.016288, 0.532147, 0.278863, 0.290231, -0.109099, -0.144300, -0.003078],
    }
)
positions = np.arange(len(stability))
figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
axis.barh(positions + 0.19, stability["fit"], height=0.36, color=BLUE, label="Fit period")
axis.barh(positions - 0.19, stability["validation"], height=0.36, color=ORANGE, label="Validation period")
axis.axvline(0, color="#222222", linewidth=0.8)
axis.set_yticks(positions, stability["feature"])
axis.invert_yaxis()
axis.set(title="Spearman association with rent changes across time", xlabel="Spearman rank correlation")
axis.legend(frameon=False)
save(figure, "association_stability.png")


# Figure 2: nonlinear dependence estimates from Experiment 13.
mutual_information = pd.Series(
    {
        "town × flat-type historical mean": 0.360833,
        "approval date number": 0.228769,
        "COE premium mean": 0.217879,
        "24-month global target lag": 0.194674,
        "street": 0.130312,
        "floor area": 0.097541,
        "block": 0.084781,
        "flat type": 0.082298,
        "nearest MRT name": 0.062681,
        "maximum floor": 0.056854,
        "year completed": 0.050954,
        "town": 0.043132,
        "nearest monument name": 0.041378,
        "nearest attraction name": 0.040233,
        "nearest mall distance": 0.038381,
        "flat model": 0.035866,
        "nearest MRT distance": 0.027314,
        "approval month": 0.002544,
    }
).sort_values()
figure, axis = plt.subplots(figsize=(9.5, 6.6), constrained_layout=True)
axis.barh(mutual_information.index, mutual_information.values, color=BLUE, alpha=0.88)
axis.set(title="Nonlinear dependence with rent in the fit period", xlabel="Estimated mutual information")
save(figure, "nonlinear_associations.png")


# Figure 3: model-dependent error EDA recorded by Experiment 15.
thresholds = pd.Series({"$250": 64.81, "$500": 38.91, "$750": 21.74, "$1,000": 10.76, "$1,500": 1.58})
segment_rate = pd.Series({"2 room": 7.37, "3 room": 6.04, "4 room": 11.24, "5 room": 12.45, "executive": 16.06})

figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
axes[0].bar(thresholds.index, thresholds.values, color=RED, alpha=0.84)
axes[0].set(title="OOF rows exceeding absolute-error thresholds", xlabel="Absolute residual threshold (SGD)", ylabel="Rows (%)")
axes[1].bar(segment_rate.index, segment_rate.values, color=ORANGE, alpha=0.88)
axes[1].set(title="Top-decile error rate by flat type", xlabel="Normalized flat type", ylabel="Rows in global top error decile (%)")
axes[1].tick_params(axis="x", rotation=20)
save(figure, "residual_profile.png")

print(f"Generated three notebook-backed figures in {FIGURE_DIR}")
