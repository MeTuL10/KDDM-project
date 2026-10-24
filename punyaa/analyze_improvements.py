"""Summarize saved improvement predictions, including a frozen-weight time check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .improve import regression_metrics, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/punyaa_improvements"))
    parser.add_argument("--selection-period", default="2024-05")
    parser.add_argument("--check-period", default="2024-10")
    args = parser.parse_args()
    primary = args.output_dir / args.selection_period
    selected = json.loads((primary / "metrics.json").read_text())
    predictions = pd.read_csv(primary / "validation_predictions.csv")
    raw = pd.read_csv(args.data_dir / "train.csv")
    for column in ("TOWN", "FLAT_TYPE"):
        predictions[column] = raw.iloc[predictions["row_id"]][column].to_numpy()
    predictions["rent_band"] = pd.cut(
        predictions["actual"], [0, 2000, 3000, 4000, np.inf],
        labels=["up to 2000", "2000 to 3000", "3000 to 4000", "above 4000"],
    )
    summaries = []
    for column in ("approval_month", "TOWN", "FLAT_TYPE", "rent_band"):
        for group, rows in predictions.groupby(column, observed=True):
            before = regression_metrics(rows["actual"].to_numpy(), rows["v2_frozen_blend"].to_numpy())
            after = regression_metrics(rows["actual"].to_numpy(), rows["selected_blend"].to_numpy())
            summaries.append({
                "dimension": column, "group": str(group), "rows": len(rows),
                "baseline_rmse": before["rmse"], "candidate_rmse": after["rmse"],
                "rmse_reduction": before["rmse"] - after["rmse"],
                "baseline_mae": before["mae"], "candidate_mae": after["mae"],
                "baseline_bias": before["bias_prediction_minus_actual"],
                "candidate_bias": after["bias_prediction_minus_actual"],
            })
    pd.DataFrame(summaries).to_csv(primary / "segment_diagnostics.csv", index=False)

    # Resample complete validation months, preserving the paired model errors.
    predictions["baseline_squared_error"] = (predictions["actual"] - predictions["v2_frozen_blend"]) ** 2
    predictions["candidate_squared_error"] = (predictions["actual"] - predictions["selected_blend"]) ** 2
    months = predictions.groupby("approval_month").agg(
        rows=("actual", "size"), baseline_sse=("baseline_squared_error", "sum"),
        candidate_sse=("candidate_squared_error", "sum"),
    )
    rng = np.random.default_rng(5228)
    blocks = rng.integers(0, len(months), size=(3000, len(months)))
    count = months["rows"].to_numpy()[blocks].sum(axis=1)
    difference = (
        np.sqrt(months["baseline_sse"].to_numpy()[blocks].sum(axis=1) / count)
        - np.sqrt(months["candidate_sse"].to_numpy()[blocks].sum(axis=1) / count)
    )
    payload = {
        "selection_period": args.selection_period,
        "selected_candidate": selected["selected_candidate"],
        "selected_weight": selected["selected_candidate_weight"],
        "primary_rmse_reduction": selected["rmse_improvement"],
        "month_block_bootstrap": {
            "draws": 3000, "seed": 5228, "months": len(months),
            "percentile_95_interval_rmse_reduction": np.quantile(difference, [0.025, 0.975]).tolist(),
            "note": "Exploratory paired resampling of validation months; ignores candidate-selection uncertainty and dependence between months. Not a significance claim.",
        },
    }
    secondary = args.output_dir / args.check_period / "validation_predictions.csv"
    if secondary.exists() and selected["selected_candidate"]:
        later = pd.read_csv(secondary)
        alpha = selected["selected_candidate_weight"]
        candidate = selected["selected_candidate"]
        frozen = (1 - alpha) * later["v2_frozen_blend"].to_numpy() + alpha * later[candidate].to_numpy()
        before = regression_metrics(later["actual"].to_numpy(), later["v2_frozen_blend"].to_numpy())
        after = regression_metrics(later["actual"].to_numpy(), frozen)
        payload["frozen_weight_check"] = {
            "validation_start": args.check_period, "rows": len(later),
            "baseline": before, "candidate": after,
            "rmse_reduction": before["rmse"] - after["rmse"],
            "note": "Blend weight frozen from primary run. Models are refitted at this later cutoff. Period overlaps the primary selection holdout and is not independent.",
        }
    save_json(args.output_dir / "analysis.json", payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved segment diagnostics to {primary / 'segment_diagnostics.csv'}")


if __name__ == "__main__":
    main()
