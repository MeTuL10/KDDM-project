"""Extrapolation experiments: piecewise-linear leaves, monotone constraints, calibration.

The test months run from 2025-03 to 2026-07, entirely beyond the last labelled
month, so every tree model predicts a level it has already seen.  This module
asks whether that is the binding constraint.  It is kept separate from
``improve.py`` on purpose: that module's checkpoint fingerprint hashes its own
source, so editing it would invalidate every cached v2/v3 fit.

Commands::

    python3 -m punyaa.extrapolation --validation-start 2024-05
    python3 -m punyaa.extrapolation --validation-start 2024-10
    python3 -m punyaa.extrapolation --calibration-only
    python3 -m punyaa.extrapolation --cross-window
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, TARGET, CausalHistoryTransformer, StaticFeatureBuilder
from .improve import bounded_blend, fit_experiment, regression_metrics, save_json
from .modeling import lightgbm_variant_parameters
from .train import _prepare_partition


#: Signs a domain expert would assert without looking at the target.
MONOTONE_CORE = {
    "FLOOR_AREA_SQM": 1,
    "approval_date_num": 1,
    "monthly_target_lag": 1,
    "town_flat_type_history_mean": 1,
    "property_flat_type_history_mean": 1,
    "median_rent_benchmark": 1,
}

#: Adds amenity proximity, where the sign is defensible but confounded by region.
MONOTONE_EXTENDED = {
    **MONOTONE_CORE,
    "mrt_dist_nearest_km": -1,
    "mall_dist_nearest_km": -1,
    "YEAR_COMPLETED": 1,
}


def monotone_constraints(columns, directions: dict[str, int]) -> list[int]:
    """Map a column order onto a LightGBM constraint vector.

    Categorical splits have no ordering, so they are always left unconstrained;
    LightGBM rejects a non-zero constraint on a categorical feature.
    """
    columns = list(columns)
    unknown = set(directions) - set(columns)
    if unknown - {"median_rent_benchmark"}:
        raise ValueError(f"Constrained columns missing from the frame: {sorted(unknown)}")
    return [
        0 if column in CATEGORICAL_FEATURES else int(directions.get(column, 0))
        for column in columns
    ]


def extrapolation_specs(reference: dict, columns, threads: int) -> dict[str, dict]:
    """Candidates that change how a leaf predicts, not which features it sees."""
    detailed = {**lightgbm_variant_parameters()["detailed"], "random_state": 42, "n_jobs": threads}
    core = monotone_constraints(columns, MONOTONE_CORE)
    extended = monotone_constraints(columns, MONOTONE_EXTENDED)
    return {
        "lightgbm_detailed": {
            "family": "lightgbm",
            "parameters": detailed,
            "fixed_iterations": reference["best_iterations"]["lightgbm_detailed"],
        },
        "lightgbm_linear_tree": {
            "family": "lightgbm",
            "parameters": {**detailed, "linear_tree": True, "linear_lambda": 1.0},
        },
        "lightgbm_linear_tree_reg": {
            "family": "lightgbm",
            "parameters": {
                **detailed, "linear_tree": True, "linear_lambda": 10.0,
                "num_leaves": 31, "max_depth": 6, "min_child_samples": 100,
            },
        },
        "lightgbm_monotone_core": {
            "family": "lightgbm",
            "parameters": {
                **detailed, "monotone_constraints": core,
                "monotone_constraints_method": "advanced",
            },
        },
        "lightgbm_monotone_extended": {
            "family": "lightgbm",
            "parameters": {
                **detailed, "monotone_constraints": extended,
                "monotone_constraints_method": "advanced",
            },
        },
        "lightgbm_linear_tree_monotone": {
            "family": "lightgbm",
            "parameters": {
                **detailed, "linear_tree": True, "linear_lambda": 1.0,
                "monotone_constraints": core,
                "monotone_constraints_method": "advanced",
            },
        },
    }


def frozen_v2_blend(reference: dict, directory: Path) -> np.ndarray:
    """Rebuild the v2 ensemble from its stored component predictions."""
    missing = [
        name for name in reference["weights"] if not (directory / f"{name}.npz").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing cached components in {directory}: {missing}")
    return sum(
        weight * np.load(directory / f"{name}.npz", allow_pickle=False)["predictions"]
        for name, weight in reference["weights"].items()
    )


def horizon_diagnostics(months: pd.Series, actual: np.ndarray, predictions: dict) -> pd.DataFrame:
    """RMSE and signed bias per validation month, ordered by forecast horizon.

    If extrapolation were the binding constraint, a candidate that can
    extrapolate would close the gap to the control as the horizon grows.
    """
    frame = pd.DataFrame({"month": np.asarray(months), "actual": actual})
    rows = []
    for horizon, month in enumerate(sorted(frame.month.unique()), 1):
        mask = (frame.month == month).to_numpy()
        row = {"month": month, "horizon": horizon, "rows": int(mask.sum()),
               "actual_mean": float(frame.actual[mask].mean())}
        for name, prediction in predictions.items():
            error = np.asarray(prediction)[mask] - frame.actual.to_numpy()[mask]
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
            row[f"{name}_bias"] = float(np.mean(error))
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_summary(actual: np.ndarray, prediction: np.ndarray) -> dict:
    """Affine and level-shift recalibration fitted on this window's own rows."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("actual and prediction must be matching non-empty vectors")
    slope, intercept = (float(value) for value in np.polyfit(prediction, actual, 1))
    bias = float(np.mean(prediction - actual))

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - actual) ** 2)))

    return {
        "rmse": rmse(prediction),
        "bias_prediction_minus_actual": bias,
        "affine_slope": slope,
        "affine_intercept": intercept,
        "prediction_sd": float(prediction.std()),
        "actual_sd": float(actual.std()),
        "rmse_after_level_shift": rmse(prediction - bias),
        "rmse_after_affine": rmse(slope * prediction + intercept),
    }


def forward_transfer(
    months: pd.Series, actual: np.ndarray, prediction: np.ndarray
) -> dict:
    """Fit the correction on the earlier half of the window, apply to the later half.

    An in-window recalibration gain is unusable unless it survives this test.
    """
    order = sorted(pd.Series(months).unique())
    split = len(order) // 2 + 1
    early = pd.Series(months).isin(order[:split]).to_numpy()
    late = ~early
    if not early.any() or not late.any():
        raise ValueError("The window needs at least two distinct months")
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    slope, intercept = np.polyfit(prediction[early], actual[early], 1)
    bias = float(np.mean(prediction[early] - actual[early]))

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - actual[late]) ** 2)))

    base = rmse(prediction[late])
    return {
        "fitted_months": order[:split],
        "applied_months": order[split:],
        "fitted_slope": float(slope),
        "fitted_bias": bias,
        "late_rmse": base,
        "late_rmse_after_affine": rmse(slope * prediction[late] + intercept),
        "late_rmse_after_level_shift": rmse(prediction[late] - bias),
        "affine_gain": base - rmse(slope * prediction[late] + intercept),
        "level_shift_gain": base - rmse(prediction[late] - bias),
    }


def cross_window_transfer(
    source_actual, source_prediction, target_actual, target_prediction
) -> dict:
    """Fit affine and level-shift corrections on one window and apply them to another.

    A correction that only works on the window it was fitted on is a property of
    that cutoff, not of the model.
    """
    arrays = [np.asarray(a, dtype=float) for a in
              (source_actual, source_prediction, target_actual, target_prediction)]
    source_y, source_p, target_y, target_p = arrays
    if source_y.shape != source_p.shape or target_y.shape != target_p.shape:
        raise ValueError("Each window needs matching actual and prediction vectors")
    if not len(source_y) or not len(target_y):
        raise ValueError("Both windows must be non-empty")
    slope, intercept = np.polyfit(source_p, source_y, 1)
    bias = float(np.mean(source_p - source_y))

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - target_y) ** 2)))

    base = rmse(target_p)
    affine = rmse(slope * target_p + intercept)
    shifted = rmse(target_p - bias)
    return {
        "source_slope": float(slope), "source_bias": bias, "target_rmse": base,
        "affine_rmse": affine, "level_shift_rmse": shifted,
        "affine_change": affine - base, "level_shift_change": shifted - base,
    }


def best_spread_scale(actual, prediction, scales=None) -> dict:
    """Best rescaling of the prediction spread around its own mean, by grid search.

    The scale is fitted on the same rows it is scored on, so the gain is an upper
    bound on what a dispersion correction could deliver.
    """
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or not len(actual):
        raise ValueError("actual and prediction must be matching non-empty vectors")
    scales = np.arange(0.95, 1.12, 0.005) if scales is None else np.asarray(scales, dtype=float)
    center = prediction.mean()

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean((values - actual) ** 2)))

    base = rmse(prediction)
    best_rmse, best_scale = min((rmse(center + s * (prediction - center)), float(s)) for s in scales)
    return {"base_rmse": base, "best_scale": best_scale, "best_rmse": best_rmse,
            "change": best_rmse - base}


def _split(data_dir: Path, validation_start: str):
    raw = pd.read_csv(data_dir / "train.csv")
    dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
    fit_mask = (dates < pd.Period(validation_start, freq="M").to_timestamp()).to_numpy()
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("validation-start must split the labelled rows")
    raw_fit = raw.loc[fit_mask].reset_index(drop=True)
    raw_valid = raw.loc[~fit_mask].reset_index(drop=True)
    x_fit, y_fit, x_valid = _prepare_partition(
        StaticFeatureBuilder(data_dir), CausalHistoryTransformer(24), raw_fit, raw_valid, "core"
    )
    return raw_valid, x_fit, y_fit, x_valid, raw_valid[TARGET].astype(float).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/punyaa_extrapolation"))
    parser.add_argument(
        "--component-dir", type=Path, default=Path("outputs/punyaa_improvements"),
        help="Where improve.py stored the v2 component predictions",
    )
    parser.add_argument(
        "--reference-metrics", type=Path, default=Path("outputs/punyaa_v2_validation_metrics.json")
    )
    parser.add_argument("--validation-start", default="2024-05")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--calibration-only", action="store_true",
        help="Skip the model fits and only recalibrate the stored v2 blend",
    )
    parser.add_argument(
        "--cross-window", action="store_true",
        help="Fit the correction on the 2024-05 window and apply it to 2024-10, and the reverse",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if args.cross_window:
        reference = json.loads(args.reference_metrics.read_text())
        raw = pd.read_csv(args.data_dir / "train.csv")
        dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
        windows = {}
        for start in ("2024-05", "2024-10"):
            mask = (dates >= pd.Period(start, freq="M").to_timestamp()).to_numpy()
            windows[start] = (
                raw.loc[mask, TARGET].to_numpy(dtype=float),
                frozen_v2_blend(reference, args.component_dir / start),
            )
        report = {"transfer": {}, "spread_scale": {}}
        for source, target in (("2024-05", "2024-10"), ("2024-10", "2024-05")):
            result = cross_window_transfer(*windows[source], *windows[target])
            report["transfer"][f"fit_{source}_apply_{target}"] = result
            print(f"fit {source} -> apply {target}: base {result['target_rmse']:.3f}, "
                  f"affine {result['affine_change']:+.3f}, level shift {result['level_shift_change']:+.3f}")
        for start, (actual, blend) in windows.items():
            result = best_spread_scale(actual, blend)
            report["spread_scale"][start] = result
            print(f"{start}: best spread scale {result['best_scale']:.3f}, change {result['change']:+.3f}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "cross_window_calibration.json", report)
        return

    reference = json.loads(args.reference_metrics.read_text())
    raw_valid, x_fit, y_fit, x_valid, y_valid = _split(args.data_dir, args.validation_start)
    target = y_valid.to_numpy(dtype=float)
    months = raw_valid["RENT_APPROVAL_DATE"]
    directory = args.output_dir / args.validation_start
    directory.mkdir(parents=True, exist_ok=True)
    print(
        f"{args.validation_start}: {len(x_fit):,} fitting rows, {len(x_valid):,} validation rows, "
        f"{len(x_fit.columns)} features",
        flush=True,
    )

    predictions: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    if not args.calibration_only:
        specs = extrapolation_specs(reference, x_fit.columns, args.threads)
        if args.experiments:
            unknown = set(args.experiments) - set(specs)
            if unknown:
                parser.error(f"Unknown experiments: {sorted(unknown)}")
            specs = {name: specs[name] for name in args.experiments}
        for name, spec in specs.items():
            started = time.time()
            prediction, best = fit_experiment(spec, x_fit, y_fit, x_valid, y_valid)
            metrics = regression_metrics(target, prediction)
            results[name] = {
                "metrics": metrics, "best_iteration": best,
                "seconds": round(time.time() - started, 1),
                "parameters": {
                    key: value for key, value in spec["parameters"].items()
                    if key in {"linear_tree", "linear_lambda", "num_leaves", "max_depth",
                               "min_child_samples", "monotone_constraints_method"}
                },
            }
            np.savez_compressed(directory / f"{name}.npz", predictions=prediction)
            predictions[name] = prediction
            print(
                f"  {name:32s} rmse={metrics['rmse']:9.4f} mae={metrics['mae']:8.3f} "
                f"iters={best:4d} {results[name]['seconds']:6.1f}s",
                flush=True,
            )

    blend = frozen_v2_blend(reference, args.component_dir / args.validation_start)
    baseline = regression_metrics(target, blend)
    print(f"\nfrozen v2 blend: rmse={baseline['rmse']:.4f} "
          f"bias={baseline['bias_prediction_minus_actual']:+.3f}", flush=True)

    blend_rows = []
    for name, prediction in predictions.items():
        if name == "lightgbm_detailed":
            continue
        weight, blended = bounded_blend(target, blend, prediction)
        metrics = regression_metrics(target, blended)
        blend_rows.append({
            "candidate": name,
            "standalone_rmse": results[name]["metrics"]["rmse"],
            "weight": weight,
            "blend_rmse": metrics["rmse"],
            "blend_mae": metrics["mae"],
            "gain": baseline["rmse"] - metrics["rmse"],
        })
    if blend_rows:
        table = pd.DataFrame(blend_rows).sort_values("blend_rmse")
        print("\n" + table.round(4).to_string(index=False), flush=True)
        table.to_csv(directory / "blend_table.csv", index=False)

    calibration = calibration_summary(target, blend)
    transfer = forward_transfer(months, target, blend)
    print(f"\ncalibration: slope={calibration['affine_slope']:.4f} "
          f"bias={calibration['bias_prediction_minus_actual']:+.2f} "
          f"affine_rmse={calibration['rmse_after_affine']:.4f}", flush=True)
    print(f"forward transfer: late {transfer['late_rmse']:.4f} -> "
          f"affine {transfer['late_rmse_after_affine']:.4f} "
          f"({transfer['affine_gain']:+.3f}), level shift "
          f"{transfer['late_rmse_after_level_shift']:.4f} "
          f"({transfer['level_shift_gain']:+.3f})", flush=True)

    diagnostics = horizon_diagnostics(
        months, target, {"v2_blend": blend, **predictions}
    )
    diagnostics.to_csv(directory / "horizon_diagnostics.csv", index=False)
    save_json(directory / "metrics.json", {
        "validation_start": args.validation_start,
        "validation_rows": int(len(target)),
        "feature_count": int(len(x_fit.columns)),
        "baseline_metrics": baseline,
        "experiments": results,
        "blend_candidates": blend_rows,
        "calibration": calibration,
        "forward_transfer": transfer,
    })
    print(f"\nWrote {directory}/metrics.json, horizon_diagnostics.csv", flush=True)


if __name__ == "__main__":
    main()
