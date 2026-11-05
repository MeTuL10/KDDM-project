"""Rebuild the block-premium map used in the final paper (report/final/figures/fig_block_map.pdf).

Run from the repository root (the folder that contains data/ and report/):

    python3 report/final/make_report_figures.py

Inputs
  data/train.csv                      course training file (rent per rental)
  data/auxiliary/sg-hdb-block.csv     course HDB block file (block coordinates)
Outputs (default folder report/final/figures)
  fig_block_map.pdf / .png            Fig. block premium map

Nothing here feeds any model; the map is descriptive.
"""
import argparse, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, GRID, BLUE, ORANGE = "#1a1a19", "#e3e2dd", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.5, "font.family": "DejaVu Sans"})


def normalise(text):
    """Lower-case and unify separators so train and block-file names join (as in the main pipeline)."""
    return re.sub(r"\s+", " ", re.sub(r"[-_/]", " ", str(text).lower().strip()))


# --------------------------------------------------------------------------- map
def block_premiums(train, blocks, shrink=10, min_rentals=5):
    """Shrunken mean gap between a block's rentals and the same-flat-type, same-month average rent."""
    train = train.copy()
    train["flat_type"] = train.FLAT_TYPE.map(normalise)
    train["log_rent"] = np.log(train.MONTHLY_RENT)
    month_mean = train.groupby(["flat_type", "RENT_APPROVAL_DATE"]).log_rent.transform("mean")
    train["gap"] = train.log_rent - month_mean
    keys = ["TOWN_n", "BLOCK_n", "STREET_n"]
    per_block = train.groupby(keys).gap.agg(["mean", "size"]).reset_index()
    per_block["premium"] = per_block["mean"] * per_block["size"] / (per_block["size"] + shrink)
    kept = per_block[per_block["size"] >= min_rentals]
    coords = blocks[keys + ["LATITUDE", "LONGITUDE"]].drop_duplicates(keys)
    return kept.merge(coords, on=keys, how="inner"), per_block, train


def make_map(data_dir, out_dir):
    train = pd.read_csv(data_dir / "train.csv")
    blocks = pd.read_csv(data_dir / "auxiliary" / "sg-hdb-block.csv")
    for frame in (train, blocks):
        for column in ("TOWN", "BLOCK", "STREET"):
            frame[column + "_n"] = frame[column].map(normalise)
    plotted, per_block, train = block_premiums(train, blocks)
    all_blocks = blocks.drop_duplicates(["TOWN_n", "BLOCK_n", "STREET_n"])
    rest = all_blocks.merge(plotted[["TOWN_n", "BLOCK_n", "STREET_n"]], on=["TOWN_n", "BLOCK_n", "STREET_n"],
                            how="left", indicator=True)
    rest = rest[rest["_merge"] == "left_only"]
    print("map: %d blocks coloured, %d grey (under five rentals or never rented) of %d HDB blocks; "
          "coloured blocks hold %.1f%% of rentals; 5th/95th percentile %.3f/%.3f"
          % (len(plotted), len(rest), len(all_blocks), 100 * plotted["size"].sum() / len(train),
             *plotted.premium.quantile([0.05, 0.95])))

    low, high = np.log(0.8), np.log(1.2)          # colour scale runs from 20% below to 20% above similar flats
    cmap = LinearSegmentedColormap.from_list("blue_orange", [BLUE, "#b9b9b4", ORANGE])
    fig, ax = plt.subplots(figsize=(3.4, 1.85))
    ordered = plotted.reindex(plotted.premium.abs().sort_values().index)   # strongest premiums drawn on top
    ax.scatter(rest.LONGITUDE, rest.LATITUDE, s=3.2, c="#e3e3df", linewidths=0, rasterized=True, zorder=1)
    scatter = ax.scatter(ordered.LONGITUDE, ordered.LATITUDE, c=ordered.premium, cmap=cmap,
                         norm=TwoSlopeNorm(0, low, high), s=3.2, linewidths=0, rasterized=True, zorder=2)
    ax.set_aspect(1 / np.cos(np.deg2rad(1.35)))
    ax.set_xlim(all_blocks.LONGITUDE.min() - 0.01, all_blocks.LONGITUDE.max() + 0.01)
    ax.set_ylim(all_blocks.LATITUDE.min() - 0.008, all_blocks.LATITUDE.max() + 0.008)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=40,
                       ticks=[np.log(0.8), np.log(0.9), 0, np.log(1.1), np.log(1.2)])
    bar.ax.set_xticklabels(["-20%", "-10%", "0", "+10%", "+20%"])
    bar.ax.tick_params(labelsize=6, length=2, width=0.4); bar.outline.set_linewidth(0.4)
    bar.set_label("Block rent compared with similar flats", fontsize=6.3, labelpad=1.5)
    fig.savefig(out_dir / "fig_block_map.pdf", bbox_inches="tight", pad_inches=0.02, dpi=450)
    fig.savefig(out_dir / "fig_block_map.png", dpi=170, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="repository root (contains data/ and report/)")
    parser.add_argument("--out", default="report/final/figures")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    make_map(root / "data", out_dir)


if __name__ == "__main__":
    main()
