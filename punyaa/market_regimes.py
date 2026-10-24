"""Backtest the pipeline in rising, reversing and flat markets.

The two validation windows used elsewhere (W1 from 2024-05, W2 from 2024-10)
both sit in a flat market, while the Kaggle test months run up to 16 months past
the last label.  This module repeats the evaluation at three earlier forecast
origins whose following twelve months differ sharply in how much rents moved,
and asks two questions:

1. Does the existing feature set follow a moving market, and do the leaf-model
   changes from ``extrapolation.py`` help when it does not?
2. Does deflating the target by a monthly market index, and projecting that
   index forward, repair the level error?

The index is a hedonic month effect: a sparse least-squares fit of log rent on
month, property, flat type, flat model and log floor area.  Deflating the target
by it leaves a cross-sectional problem the trees handle well, and the remaining
risk is concentrated in the projected index, which is fixed a priori (no tuning
on any validation window).

Commands::

    python3 -m punyaa.market_regimes
    python3 -m punyaa.market_regimes --families catboost --output-dir outputs/punyaa_regimes/catboost
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from .extrapolation import MONOTONE_CORE, monotone_constraints
from .features import TARGET, CausalHistoryTransformer, StaticFeatureBuilder, normalize_category
from .improve import fit_experiment, save_json
from .modeling import catboost_parameters, lightgbm_variant_parameters
from .noise_floor import noise_estimate, noise_interval
from .train import _prepare_partition


#: (first validation month, last validation month) of each twelve-month backtest.
ORIGINS = (("2022-01", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-12"))

#: Features that carry the observed rent level through earlier labels.
HISTORY_FEATURES = [
    "monthly_target_lag",
    "town_flat_type_history_mean",
    "town_flat_type_history_count",
    "property_flat_type_history_mean",
    "property_flat_type_history_count",
]

#: Features that tell a tree which calendar period a row belongs to.
TIME_FEATURES = ["approval_date_num", "approval_month", "coe_quota_premium_mean"]

PROJECTION_MODES = ("flat", "linear", "damped")
EPOCH_YEAR = 2021


def month_number(dates) -> np.ndarray:
    """Whole months since January of the first labelled year (2021-01 is 0)."""
    periods = pd.PeriodIndex(pd.Series(dates), freq="M")
    return ((periods.year - EPOCH_YEAR) * 12 + periods.month - 1).to_numpy()


def month_index(frame: pd.DataFrame, anchor_months) -> pd.Series:
    """Hedonic month effects on log rent, normalised to zero over ``anchor_months``.

    The design matrix holds one indicator block each for month, property
    (town, block, street), flat type and flat model, plus a centred log floor
    area column.  A small ridge term (``damp``) pins down the otherwise
    unidentified block offsets, and the month effects are re-centred on the
    anchor, so only differences between months carry meaning.
    """
    months = month_number(frame["RENT_APPROVAL_DATE"])
    anchor_months = list(anchor_months)
    if not set(anchor_months) <= set(months):
        raise ValueError("Anchor months must all occur in the frame")
    property_key = (
        normalize_category(frame["TOWN"]) + "|" + normalize_category(frame["BLOCK"])
        + "|" + normalize_category(frame["STREET"])
    ).to_numpy()
    columns = [
        months, property_key,
        normalize_category(frame["FLAT_TYPE"]).to_numpy(),
        normalize_category(frame["FLAT_MODEL"]).to_numpy(),
    ]
    rent = frame[TARGET].to_numpy(dtype=float)
    if (rent <= 0).any():
        raise ValueError("Rents must be positive to take logs")
    target = np.log(rent)
    rows = np.arange(len(frame))
    blocks, month_levels = [], None
    for position, column in enumerate(columns):
        codes, levels = pd.factorize(column, sort=(position == 0))
        blocks.append(sp.csr_matrix(
            (np.ones(len(frame)), (rows, codes)), shape=(len(frame), len(levels))
        ))
        if position == 0:
            month_levels = levels
    area = np.log(frame["FLOOR_AREA_SQM"].to_numpy(dtype=float))
    blocks.append(sp.csr_matrix((area - area.mean())[:, None]))
    solution = lsqr(sp.hstack(blocks).tocsr(), target - target.mean(), damp=1e-3, iter_lim=3000)[0]
    index = pd.Series(solution[: len(month_levels)], index=np.asarray(month_levels))
    return index - index.loc[anchor_months].mean()


def tail_slope(index: pd.Series, window: int = 6) -> float:
    """Least-squares slope of the last ``window`` index values, per month."""
    tail = index.sort_index().iloc[-window:]
    if len(tail) < 2:
        raise ValueError("Need at least two months to estimate a slope")
    return float(np.polyfit(tail.index.to_numpy(dtype=float), tail.to_numpy(), 1)[0])


def project_index(
    index: pd.Series, months, mode: str, *, window: int = 6, damp: float = 0.9
) -> np.ndarray:
    """Log-index adjustment for ``months``, extending ``index`` past its last month.

    ``flat`` holds the last level (zero adjustment, because the index is anchored
    on its final months).  ``linear`` continues the recent slope forever.
    ``damped`` continues it with geometrically shrinking steps, so the total
    drift is bounded by ``slope * damp / (1 - damp)``.  Months already inside the
    index get zero, because their deflation uses the fitted index directly.
    """
    if mode not in PROJECTION_MODES:
        raise ValueError(f"Unknown projection mode {mode!r}; expected one of {PROJECTION_MODES}")
    if not 0.0 < damp < 1.0:
        raise ValueError("damp must lie strictly between 0 and 1")
    horizon = np.maximum(np.asarray(months) - index.index.max(), 0).astype(float)
    if mode == "flat":
        return np.zeros(len(horizon))
    slope = tail_slope(index, window)
    if mode == "linear":
        return slope * horizon
    return slope * damp * (1.0 - damp**horizon) / (1.0 - damp)


def window_masks(dates: pd.Series, first_month: str, last_month: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit rows lie strictly before ``first_month``; validation rows span the window."""
    start = pd.Period(first_month, freq="M").to_timestamp()
    stop = pd.Period(last_month, freq="M").to_timestamp() + pd.offsets.MonthBegin(1)
    if stop <= start:
        raise ValueError("The window must end no earlier than it starts")
    fit = (dates < start).to_numpy()
    validation = ((dates >= start) & (dates < stop)).to_numpy()
    if not fit.any() or not validation.any():
        raise ValueError("The window leaves no fitting or no validation rows")
    return fit, validation


def error_summary(prediction: np.ndarray, actual: np.ndarray, months: np.ndarray) -> dict:
    """RMSE and signed bias overall and in the first and last three months."""
    error = np.asarray(prediction, dtype=float) - np.asarray(actual, dtype=float)
    if not np.isfinite(error).all():
        raise ValueError("Non-finite predictions")
    order = sorted(set(months))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "bias_first3": float(error[np.isin(months, order[:3])].mean()),
        "bias_last3": float(error[np.isin(months, order[-3:])].mean()),
    }


def ablation_specs(columns, threads: int) -> dict[str, dict]:
    """Control and the changes tested in the flat-market chapters, for a moving market."""
    detailed = {**lightgbm_variant_parameters()["detailed"], "random_state": 42, "n_jobs": threads}
    return {
        "control": {"family": "lightgbm", "parameters": detailed},
        "no_24_month_lag": {
            "family": "lightgbm", "parameters": detailed, "drop_features": ["monthly_target_lag"],
        },
        "no_rent_history": {
            "family": "lightgbm", "parameters": detailed, "drop_features": HISTORY_FEATURES,
        },
        "linear_leaves": {
            "family": "lightgbm",
            "parameters": {
                **detailed, "linear_tree": True, "linear_lambda": 10.0,
                "num_leaves": 31, "max_depth": 6, "min_child_samples": 100,
            },
        },
        "monotone_core": {
            "family": "lightgbm",
            "parameters": {
                **detailed,
                "monotone_constraints": monotone_constraints(columns, MONOTONE_CORE),
                "monotone_constraints_method": "advanced",
            },
        },
    }


def family_parameters(family: str, threads: int) -> dict:
    if family == "lightgbm":
        return {**lightgbm_variant_parameters()["detailed"], "random_state": 42, "n_jobs": threads}
    if family == "catboost":
        return {**catboost_parameters(), "random_seed": 42, "thread_count": threads}
    raise ValueError(f"Unknown family {family!r}")


def run_origin(
    raw: pd.DataFrame, builder: StaticFeatureBuilder, first_month: str, last_month: str, *,
    threads: int, families: tuple[str, ...], stages: tuple[str, ...],
) -> tuple[list[dict], dict]:
    """All models for one forecast origin; returns result rows and a context summary."""
    dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
    fit_mask, validation_mask = window_masks(dates, first_month, last_month)
    raw_fit = raw[fit_mask].reset_index(drop=True)
    raw_valid = raw[validation_mask].reset_index(drop=True)
    x_fit, y_fit, x_valid = _prepare_partition(
        builder, CausalHistoryTransformer(24), raw_fit, raw_valid, "core"
    )
    y_valid = raw_valid[TARGET].astype(float).reset_index(drop=True)
    actual = y_valid.to_numpy()
    valid_months = raw_valid["RENT_APPROVAL_DATE"].to_numpy()
    fit_numbers, valid_numbers = month_number(raw_fit["RENT_APPROVAL_DATE"]), month_number(raw_valid["RENT_APPROVAL_DATE"])
    anchor = sorted(set(fit_numbers))[-3:]
    fit_index = month_index(raw_fit, anchor)
    full_index = month_index(pd.concat([raw_fit, raw_valid], ignore_index=True), anchor)
    order = sorted(set(valid_months))
    first_mean = float(actual[valid_months == order[0]].mean())
    last_mean = float(actual[valid_months == order[-1]].mean())
    noise = noise_estimate(raw_valid)
    noise.update(noise_interval(raw_valid))
    context = {
        "origin": first_month, "window_end": last_month,
        "fit_rows": int(len(x_fit)), "validation_rows": int(len(x_valid)),
        "market_change_pct": 100.0 * (last_mean / first_mean - 1.0),
        "fit_mean_rent": float(y_fit.mean()),
        "lag_missing_share_in_fit": float(x_fit["monthly_target_lag"].isna().mean()),
        "true_index_drift": float(full_index.loc[valid_numbers.max()] - full_index.loc[anchor[-1]]),
        "fit_tail_slope_per_month": tail_slope(fit_index),
        "noise": noise,
    }
    print(
        f"\n=== origin {first_month}: fit {len(x_fit):,}, validation {len(x_valid):,}, "
        f"market {context['market_change_pct']:+.1f}%, true index drift "
        f"{100 * context['true_index_drift']:+.1f}%, noise SD {noise['unbiased_noise_sd']:.1f}",
        flush=True,
    )

    rows: list[dict] = []

    def record(family: str, name: str, prediction, iterations: int, seconds: float) -> None:
        summary = error_summary(prediction, actual, valid_months)
        rows.append({"origin": first_month, "family": family, "model": name, **summary,
                     "iterations": int(iterations)})
        print(
            f"  {family:9s} {name:22s} rmse {summary['rmse']:8.2f} bias {summary['bias']:+8.1f} "
            f"last3 {summary['bias_last3']:+8.1f} iters {iterations:5d} {seconds:5.0f}s", flush=True,
        )

    drop = HISTORY_FEATURES + TIME_FEATURES
    for family in families:
        parameters = family_parameters(family, threads)
        if "ablations" in stages and family == "lightgbm":
            for name, spec in ablation_specs(x_fit.columns, threads).items():
                started = time.time()
                prediction, best = fit_experiment(spec, x_fit, y_fit, x_valid, y_valid)
                record(family, name, prediction, best, time.time() - started)
        elif "ablations" in stages:
            started = time.time()
            prediction, best = fit_experiment(
                {"family": family, "parameters": parameters}, x_fit, y_fit, x_valid, y_valid
            )
            record(family, "control", prediction, best, time.time() - started)
        if "index" not in stages:
            continue
        static = {"family": family, "parameters": parameters, "drop_features": drop}
        deflated_fit = pd.Series(y_fit.to_numpy() / np.exp(fit_index.loc[fit_numbers].to_numpy()))
        runs = {}
        if family == "lightgbm":
            started = time.time()
            prediction, best = fit_experiment(static, x_fit, y_fit, x_valid, y_valid)
            record(family, "static_only", prediction, best, time.time() - started)
            for mode in PROJECTION_MODES:
                runs[f"deflated_{mode}"] = project_index(fit_index, valid_numbers, mode)
        else:
            runs["deflated_damped"] = project_index(fit_index, valid_numbers, "damped")
        runs["deflated_oracle"] = full_index.loc[valid_numbers].to_numpy()
        for name, adjustment in runs.items():
            started = time.time()
            # Early stopping sees the validation target on the same deflated scale, as elsewhere in the project.
            prediction, best = fit_experiment(
                static, x_fit, deflated_fit, x_valid, pd.Series(actual / np.exp(adjustment))
            )
            record(family, name, prediction * np.exp(adjustment), best, time.time() - started)
    return rows, context


def test_period_projection(
    raw: pd.DataFrame, *, windows=(3, 6, 12), horizons=(1, 6, 12, 16), damp: float = 0.9
) -> tuple[pd.Series, pd.DataFrame]:
    """What the projected index would add at the test horizons, given all labelled rows.

    Returns the full-data index (zero over the last three labelled months) and a
    table of the damped and linear adjustments for several slope windows.  The
    slope window is the least certain choice, so the table shows its sensitivity
    instead of fixing one value.  Adjustments are also expressed in SGD at the
    mean rent of the last three labelled months.
    """
    numbers = month_number(raw["RENT_APPROVAL_DATE"])
    anchor = sorted(set(numbers))[-3:]
    index = month_index(raw, anchor)
    level = float(raw.loc[np.isin(numbers, anchor), TARGET].mean())
    rows = []
    for window in windows:
        slope = tail_slope(index, window)
        for horizon in horizons:
            months = [index.index.max() + horizon]
            for mode in ("damped", "linear"):
                factor = float(np.exp(project_index(index, months, mode, window=window, damp=damp)[0]) - 1.0)
                rows.append({
                    "slope_window_months": window, "slope_pct_per_month": 100.0 * slope,
                    "horizon_months": horizon, "mode": mode,
                    "adjustment_pct": 100.0 * factor, "adjustment_sgd": level * factor,
                })
    return index, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/punyaa_regimes"))
    parser.add_argument(
        "--origins", nargs="+", default=[f"{a}:{b}" for a, b in ORIGINS],
        help="Windows as FIRST:LAST months, e.g. 2022-01:2022-12",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--families", nargs="+", choices=("lightgbm", "catboost"), default=["lightgbm"],
                        help="CatBoost runs the control and the damped and oracle index variants only")
    parser.add_argument("--stages", nargs="+", choices=("ablations", "index"),
                        default=["ablations", "index"])
    parser.add_argument("--test-projection", action="store_true",
                        help="Only write the full-data index and its implied test-period adjustments")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if args.test_projection:
        raw = pd.read_csv(args.data_dir / "train.csv")
        index, table = test_period_projection(raw)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        labels = [f"{EPOCH_YEAR + int(m) // 12}-{int(m) % 12 + 1:02d}" for m in index.index]
        pd.DataFrame({"month": labels, "log_index": index.to_numpy()}).to_csv(
            args.output_dir / "full_data_index.csv", index=False)
        table.to_csv(args.output_dir / "test_period_projection.csv", index=False)
        print(table.round(3).to_string(index=False), flush=True)
        return

    origins = [tuple(text.split(":")) for text in args.origins]
    if any(len(pair) != 2 for pair in origins):
        parser.error("Each origin must look like FIRST:LAST")
    families = tuple(args.families)
    raw = pd.read_csv(args.data_dir / "train.csv")
    builder = StaticFeatureBuilder(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    contexts: list[dict] = []
    for first_month, last_month in origins:
        rows, context = run_origin(
            raw, builder, first_month, last_month,
            threads=args.threads, families=families, stages=tuple(args.stages),
        )
        results.extend(rows)
        contexts.append(context)
        pd.DataFrame(results).to_csv(args.output_dir / "regime_results.csv", index=False)
        save_json(args.output_dir / "origins.json", {"origins": contexts})
    print(f"\nWrote {args.output_dir}/regime_results.csv and origins.json", flush=True)


if __name__ == "__main__":
    main()
