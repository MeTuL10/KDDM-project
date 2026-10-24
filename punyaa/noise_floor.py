"""Estimate how much rent variation the model inputs cannot explain.

Two listings that share every model input receive the same prediction from any
deterministic model, because the engineered features are functions of those
inputs and of the labelled history.  The spread of rents inside such a group is
therefore error no model of these inputs can remove.

Two quantities are reported, and they differ for a reason that matters:

* ``unbiased_noise_sd`` divides the within-group sum of squares by the degrees
  of freedom (rows minus groups).  It estimates the noise standard deviation and
  is the reference to compare a model against.
* ``realized_lower_bound`` divides by the row count.  It is a strict bound on the
  RMSE of any deterministic predictor for those exact rows, but it is loose:
  groups are small, so the group's own mean is fitted to the same rows it is
  scored on.

Command::

    python3 -m punyaa.noise_floor --validation-start 2024-05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import TARGET, normalize_category


#: Every raw input a model can use.  FURNISHED and FEE are constant in the
#: labelled data, so they cannot separate rows and are omitted.
FEATURE_KEY = [
    "RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE",
]
TEXT_COLUMNS = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]


def _keyed(frame: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    missing = [column for column in [*key, TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[[*key, TARGET]].copy()
    for column in TEXT_COLUMNS:
        if column in key:
            result[column] = normalize_category(result[column])
    return result


def twin_mask(frame: pd.DataFrame, key: list[str] = FEATURE_KEY) -> np.ndarray:
    """True for rows that share every key column with at least one other row."""
    keyed = _keyed(frame, key)
    return (keyed.groupby(key, dropna=False)[TARGET].transform("size") >= 2).to_numpy()


def noise_estimate(frame: pd.DataFrame, key: list[str] = FEATURE_KEY) -> dict:
    """Summarize within-group rent spread for rows that have an exact twin."""
    keyed = _keyed(frame, key)
    grouped = keyed.groupby(key, dropna=False)[TARGET]
    size = grouped.transform("size")
    twins = size >= 2
    if not twins.any():
        raise ValueError("No two rows share every model input; no noise estimate exists")
    subset = keyed[twins]
    deviation = subset[TARGET] - grouped.transform("mean")[twins]
    sum_squares = float((deviation**2).sum())
    groups = int(subset.groupby(key, dropna=False).ngroups)
    degrees_of_freedom = len(subset) - groups
    return {
        "rows": int(len(keyed)),
        "rows_with_twin": int(twins.sum()),
        "share_with_twin": float(twins.mean()),
        "groups": groups,
        "mean_group_size": float(len(subset) / groups),
        "unbiased_noise_sd": float(np.sqrt(sum_squares / degrees_of_freedom)),
        "realized_lower_bound": float(np.sqrt(sum_squares / len(subset))),
    }


def noise_interval(
    frame: pd.DataFrame, key: list[str] = FEATURE_KEY, *, draws: int = 2000, seed: int = 5228
) -> dict:
    """Percentile interval for the noise SD, resampling whole groups with replacement.

    Rows inside a group are not independent of each other, so resampling rows
    would understate the uncertainty.
    """
    if draws < 100:
        raise ValueError("draws must be at least 100")
    keyed = _keyed(frame, key)
    grouped = keyed.groupby(key, dropna=False)[TARGET]
    twins = keyed[grouped.transform("size") >= 2]
    if twins.empty:
        raise ValueError("No two rows share every model input; no noise estimate exists")
    per_group = twins.groupby(key, dropna=False)[TARGET].agg(
        sum_squares=lambda values: float(((values - values.mean()) ** 2).sum()),
        degrees_of_freedom=lambda values: len(values) - 1,
    )
    sum_squares = per_group["sum_squares"].to_numpy()
    degrees = per_group["degrees_of_freedom"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_group), size=(draws, len(per_group)))
    estimates = np.sqrt(sum_squares[picks].sum(1) / degrees[picks].sum(1))
    return {
        "draws": draws, "seed": seed,
        "lower_95": float(np.percentile(estimates, 2.5)),
        "upper_95": float(np.percentile(estimates, 97.5)),
    }


def _rmse(prediction: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(actual)) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-start", default="2024-05")
    parser.add_argument(
        "--reference-metrics", type=Path, default=Path("outputs/punyaa_v2_validation_metrics.json")
    )
    parser.add_argument(
        "--component-dir", type=Path, default=Path("outputs/punyaa_improvements"),
        help="Where improve.py stored the v2 component predictions",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/punyaa_noise_floor"))
    args = parser.parse_args()

    raw = pd.read_csv(args.data_dir / "train.csv")
    dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
    in_window = (dates >= pd.Period(args.validation_start, freq="M").to_timestamp()).to_numpy()
    if not in_window.any() or in_window.all():
        parser.error("validation-start must split the labelled rows")
    window = raw.loc[in_window].reset_index(drop=True)

    result = {
        "validation_start": args.validation_start,
        "all_labelled_months": noise_estimate(raw),
        "validation_window": noise_estimate(window),
    }
    result["validation_window"]["interval"] = noise_interval(window)
    reference = json.loads(args.reference_metrics.read_text())
    directory = args.component_dir / args.validation_start
    if all((directory / f"{name}.npz").exists() for name in reference["weights"]):
        blend = sum(
            weight * np.load(directory / f"{name}.npz", allow_pickle=False)["predictions"]
            for name, weight in reference["weights"].items()
        )
        actual = window[TARGET].to_numpy(dtype=float)
        mask = twin_mask(window)
        result["v2_blend_rmse"] = {
            "all_rows": _rmse(blend, actual),
            "rows_with_twin": _rmse(blend[mask], actual[mask]),
            "rows_without_twin": _rmse(blend[~mask], actual[~mask]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.validation_start}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in ("all_labelled_months", "validation_window"):
        row = result[name]
        print(f"{name:20s} rows {row['rows']:>7,}  with twin {row['share_with_twin']:6.1%}  "
              f"noise SD {row['unbiased_noise_sd']:7.2f}  strict bound {row['realized_lower_bound']:7.2f}")
    interval = result["validation_window"]["interval"]
    print(f"validation noise SD 95% interval [{interval['lower_95']:.2f}, {interval['upper_95']:.2f}] "
          f"({interval['draws']} group resamples, seed {interval['seed']})")
    if "v2_blend_rmse" in result:
        print("v2 blend RMSE", {key: round(value, 3) for key, value in result["v2_blend_rmse"].items()})
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
