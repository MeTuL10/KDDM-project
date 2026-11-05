"""Report versions of two EDA figures (Fig. 1 and Fig. 3 in main.pdf), drawn in
the same matplotlib style as the results figures (Fig. 5 and Fig. 6).

Rows come from data_cleaning.R so the numbers match eda.R exactly; they are
exported once to eda/eda_rows.csv. Outputs (PDF for LaTeX, PNG for preview):
  eda/report_rent_vs_benchmark.{pdf,png}
  eda/report_box_flat_type.{pdf,png}
"""
import os
import subprocess

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

ROWS = "eda/eda_rows.csv"
BLUE, ORANGE, GREY = "#1f5fa8", "#d1691b", "#8c8c8c"
COL_W = 3.5  # IEEE column width, inches

EXPORT_R = """
suppressMessages(library(dplyr))
source("data_cleaning.R")
train_feature |>
  transmute(month = rent_approval_date,
            flat_type = ifelse(room_count == 6, "executive", paste0(room_count, "-room")),
            monthly_rent, median_rent_benchmark) |>
  write.csv("%s", row.names = FALSE)
""" % ROWS

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d0d0d0",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
comma = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"eda/{name}.{ext}")
    plt.close(fig)


if not os.path.exists(ROWS):
    subprocess.run(["Rscript", "-e", EXPORT_R], check=True)
df = pd.read_csv(ROWS)
df["month"] = pd.to_datetime(df["month"] + "-01")

# Fig. 1: mean rent and mean benchmark per approval month
monthly = df.groupby("month")[["monthly_rent", "median_rent_benchmark"]].mean()
fig, ax = plt.subplots(figsize=(COL_W, 1.9))
ax.plot(monthly.index, monthly["monthly_rent"], color=BLUE, lw=1.4, label="mean monthly rent")
ax.plot(monthly.index, monthly["median_rent_benchmark"], color=ORANGE, lw=1.4,
        label="mean town × flat-type median")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.yaxis.set_major_formatter(comma)
ax.set_xlabel("rent approval month")
ax.set_ylabel("SGD")
ax.legend(loc="lower right", handlelength=1.5)
save(fig, "report_rent_vs_benchmark")

# Fig. 3: monthly rent by flat type, 1.5 x IQR points shown
types = ["1-room", "2-room", "3-room", "4-room", "5-room", "executive"]
fig, ax = plt.subplots(figsize=(COL_W, 2.0))
ax.boxplot([df.loc[df["flat_type"] == t, "monthly_rent"] for t in types],
           tick_labels=[t.replace("executive", "exec.") for t in types], widths=0.6, whis=1.5, patch_artist=True,
           boxprops=dict(facecolor="#d5e1f0", edgecolor=BLUE, linewidth=0.8),
           whiskerprops=dict(color=BLUE, linewidth=0.8),
           capprops=dict(color=BLUE, linewidth=0.8),
           medianprops=dict(color=BLUE, linewidth=1.4),
           flierprops=dict(marker="o", markersize=1.6, markerfacecolor=ORANGE,
                           markeredgecolor="none", alpha=0.5))
ax.yaxis.set_major_formatter(comma)
ax.set_ylim(bottom=0)
ax.set_ylabel("monthly rent (SGD)")
ax.set_xlabel("flat type")
ax.tick_params(axis="x", length=0)
save(fig, "report_box_flat_type")
