"""Run reproducible improvement experiments without changing the v2 pipeline.

The common holdout is used for model selection, not as an untouched test set.
Each completed fit is checkpointed; rerunning the command reuses matching fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, TARGET, CausalHistoryTransformer, StaticFeatureBuilder
from .modeling import (
    _catboost_frame,
    _lightgbm_frames,
    catboost_parameters,
    lightgbm_variant_parameters,
    make_submission,
)
from .train import _prepare_partition


def recency_weights(months: np.ndarray, half_life: float) -> np.ndarray:
    """Exponentially decay older months, normalizing the mean weight to one."""
    months = np.asarray(months, dtype=float)
    if months.ndim != 1 or not len(months) or not np.isfinite(months).all():
        raise ValueError("months must be a non-empty finite vector")
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life must be positive")
    weights = np.exp2((months - months.max()) / half_life)
    return weights / weights.mean()


def bounded_blend(
    target: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, maximum: float = 0.5
) -> tuple[float, np.ndarray]:
    """Find the least-squares candidate weight, preserving >=50% of v2 by default."""
    target, baseline, candidate = (
        np.asarray(values, dtype=float) for values in (target, baseline, candidate)
    )
    if not (target.shape == baseline.shape == candidate.shape) or target.ndim != 1:
        raise ValueError("blend vectors must have matching one-dimensional shapes")
    if not all(np.isfinite(values).all() for values in (target, baseline, candidate)):
        raise ValueError("blend vectors must be finite")
    if not 0 <= maximum <= 1:
        raise ValueError("maximum must lie between zero and one")
    difference = candidate - baseline
    denominator = float(difference @ difference)
    weight = (
        float(np.clip((target - baseline) @ difference / denominator, 0, maximum))
        if denominator > 0
        else 0.0
    )
    return weight, baseline + weight * difference


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction) - np.asarray(target)
    if not np.isfinite(error).all():
        raise ValueError("Non-finite model predictions")
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias_prediction_minus_actual": float(np.mean(error)),
    }


def experiment_specs(reference: dict, threads: int) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for variant, parameters in lightgbm_variant_parameters().items():
        name = f"lightgbm_{variant}"
        specs[name] = {
            "family": "lightgbm",
            "parameters": {**parameters, "random_state": 42, "n_jobs": threads},
            "fixed_iterations": reference["best_iterations"][name],
        }
    for seed in (42, 2026, 5228):
        name = f"catboost_seed_{seed}"
        specs[name] = {
            "family": "catboost",
            "parameters": {
                **catboost_parameters(), "random_seed": seed, "thread_count": threads
            },
            "fixed_iterations": reference["best_iterations"][name],
        }
    detailed = specs["lightgbm_detailed"]["parameters"]
    for half_life in (12, 24):
        specs[f"lightgbm_recency_{half_life}"] = {
            "family": "lightgbm", "parameters": detailed,
            "half_life_months": half_life,
        }
        specs[f"catboost_recency_{half_life}"] = {
            "family": "catboost",
            "parameters": specs["catboost_seed_5228"]["parameters"],
            "half_life_months": half_life,
        }
    specs["lightgbm_higher_capacity"] = {
        "family": "lightgbm",
        "parameters": {
            **detailed, "num_leaves": 255, "max_depth": -1,
            "min_child_samples": 80, "reg_lambda": 8.0, "reg_alpha": 0.5,
            "colsample_bytree": 0.8,
        },
    }
    specs["lightgbm_without_property_history"] = {
        "family": "lightgbm", "parameters": detailed,
        "drop_features": [
            "property_flat_type_history_mean", "property_flat_type_history_count"
        ],
    }
    specs["lightgbm_rent_per_sqm"] = {
        "family": "lightgbm", "parameters": detailed, "target_per_sqm": True,
    }
    specs["xgboost_hist"] = {
        "family": "xgboost",
        "parameters": {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "tree_method": "hist", "enable_categorical": True,
            "n_estimators": 2200, "learning_rate": 0.035, "max_depth": 7,
            "min_child_weight": 30, "reg_lambda": 10.0, "reg_alpha": 1.0,
            "subsample": 0.85, "colsample_bytree": 0.8, "max_bin": 256,
            "max_cat_to_onehot": 4, "max_cat_threshold": 64,
            "random_state": 42, "n_jobs": threads,
        },
    }
    specs["standard_gbm"] = {
        "family": "gbm",
        "parameters": {
            "loss": "squared_error", "n_estimators": 400,
            "learning_rate": 0.05, "max_depth": 3,
            "min_samples_leaf": 30, "subsample": 0.85, "random_state": 42,
        },
    }
    return specs


def fit_experiment(
    spec: dict,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_query: pd.DataFrame,
    y_query: pd.Series | None = None,
    *,
    iterations: int | None = None,
) -> tuple[np.ndarray, int]:
    """Fit a candidate; a missing query target means a fixed-iteration final refit."""
    parameters = dict(spec["parameters"])
    fixed = iterations or spec.get("fixed_iterations")
    x_fit = x_fit.drop(columns=spec.get("drop_features", []))
    x_query = x_query.drop(columns=spec.get("drop_features", []))
    fit_scale = np.ones(len(x_fit))
    query_scale = np.ones(len(x_query))
    if spec.get("target_per_sqm"):
        fit_scale = x_fit["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        query_scale = x_query["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        if (fit_scale <= 0).any() or (query_scale <= 0).any():
            raise ValueError("Rent-per-sqm requires positive floor areas")
    fit_target = y_fit.to_numpy(dtype=float) / fit_scale
    query_target = (
        y_query.to_numpy(dtype=float) / query_scale if y_query is not None else None
    )
    weights = None
    if spec.get("half_life_months"):
        weights = recency_weights(
            x_fit["approval_date_num"].to_numpy(), spec["half_life_months"]
        )
    family = spec["family"]

    if family == "lightgbm":
        import lightgbm as lgb

        fit_frame, query_frame = _lightgbm_frames(x_fit, x_query)
        if fixed:
            parameters["n_estimators"] = int(fixed)
        fit_arguments = {"sample_weight": weights, "categorical_feature": CATEGORICAL_FEATURES}
        if query_target is not None:
            fit_arguments.update(eval_X=query_frame, eval_y=query_target)
            if spec.get("target_per_sqm"):
                parameters["metric"] = "None"

                def rent_rmse(actual, predicted):
                    return "rent_rmse", float(np.sqrt(np.mean(((actual - predicted) * query_scale) ** 2))), False

                fit_arguments["eval_metric"] = rent_rmse
            if not fixed:
                fit_arguments["callbacks"] = [lgb.early_stopping(120, verbose=False)]
        model = lgb.LGBMRegressor(**parameters)
        model.fit(fit_frame, fit_target, **fit_arguments)
        prediction = model.predict(query_frame)
        best = int(model.best_iteration_ or parameters["n_estimators"])

    elif family == "catboost":
        from catboost import CatBoostRegressor

        fit_frame, query_frame = _catboost_frame(x_fit), _catboost_frame(x_query)
        if fixed:
            parameters["iterations"] = int(fixed)
        fit_arguments = {
            "cat_features": CATEGORICAL_FEATURES, "sample_weight": weights,
            "verbose": False,
        }
        if query_target is not None:
            fit_arguments["eval_set"] = (query_frame, query_target)
            if not fixed:
                fit_arguments["early_stopping_rounds"] = 120
            else:
                fit_arguments["use_best_model"] = False
        model = CatBoostRegressor(**parameters)
        model.fit(fit_frame, fit_target, **fit_arguments)
        prediction = model.predict(query_frame)
        best = int(parameters["iterations"] if fixed else model.get_best_iteration() + 1)

    elif family == "xgboost":
        from xgboost import XGBRegressor

        fit_frame, query_frame = _lightgbm_frames(x_fit, x_query)
        if fixed:
            parameters["n_estimators"] = int(fixed)
        fit_arguments = {"sample_weight": weights, "verbose": False}
        if query_target is not None:
            fit_arguments["eval_set"] = [(query_frame, query_target)]
            if not fixed:
                parameters["early_stopping_rounds"] = 100
        model = XGBRegressor(**parameters)
        model.fit(fit_frame, fit_target, **fit_arguments)
        prediction = model.predict(query_frame)
        best = int(fixed or model.best_iteration + 1)

    elif family == "gbm":
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder

        numeric_columns = [column for column in x_fit if column not in CATEGORICAL_FEATURES]
        encoder = ColumnTransformer([
            ("categories", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        ], sparse_threshold=1.0)
        fit_frame = encoder.fit_transform(_catboost_frame(x_fit))
        query_frame = encoder.transform(_catboost_frame(x_query))
        if fixed:
            parameters["n_estimators"] = int(fixed)
        model = GradientBoostingRegressor(**parameters)
        model.fit(fit_frame, fit_target, sample_weight=weights)
        best = int(parameters["n_estimators"])
        prediction = model.predict(query_frame)
        if query_target is not None and not fixed:
            best_score = np.inf
            for stage, stage_prediction in enumerate(model.staged_predict(query_frame), 1):
                score = float(np.mean(((query_target - stage_prediction) * query_scale) ** 2))
                if score < best_score:
                    best_score, best, prediction = score, stage, stage_prediction.copy()
    else:
        raise ValueError(f"Unknown experiment family: {family}")

    return np.asarray(prediction, dtype=float) * query_scale, best


def save_json(path: Path, value: dict) -> None:
    """Write checkpoints after each successful model rather than only at the end."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/punyaa_improvements"))
    parser.add_argument("--reference-metrics", type=Path, default=Path("outputs/punyaa_v2_validation_metrics.json"))
    parser.add_argument("--validation-start", default="2024-05")
    parser.add_argument("--experiments", nargs="+", help="Optional subset of candidate names")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--submission", type=Path, help="Refit the selected blend and export its submission")
    args = parser.parse_args()

    import lightgbm, catboost, xgboost, sklearn

    reference = json.loads(args.reference_metrics.read_text())
    specs = experiment_specs(reference, args.threads)
    if args.experiments:
        unknown = set(args.experiments) - set(specs)
        if unknown:
            parser.error(f"Unknown experiments: {sorted(unknown)}")
        specs = {name: specs[name] for name in args.experiments}
    raw = pd.read_csv(args.data_dir / "train.csv")
    dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
    fit_mask = dates < pd.Period(args.validation_start, freq="M").to_timestamp()
    if not fit_mask.any() or fit_mask.all():
        parser.error("validation-start must split the labelled rows")
    raw_fit, raw_valid = raw.loc[fit_mask].reset_index(drop=True), raw.loc[~fit_mask].reset_index(drop=True)
    builder = StaticFeatureBuilder(args.data_dir)
    print(f"Preparing {len(raw_fit):,} fitting and {len(raw_valid):,} validation rows", flush=True)
    x_fit, y_fit, x_valid = _prepare_partition(
        builder, CausalHistoryTransformer(24), raw_fit, raw_valid, "core"
    )
    y_valid = raw_valid[TARGET].astype(float)
    directory = args.output_dir / args.validation_start
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in [Path(__file__), Path(__file__).with_name("features.py"), Path(__file__).with_name("modeling.py"), args.reference_metrics]:
        digest.update(path.read_bytes())
    for frame in (x_fit, x_valid, y_fit, y_valid):
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    versions = {
        "python": platform.python_version(), "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__, "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
    }
    digest.update(json.dumps(versions, sort_keys=True).encode())
    fingerprint = digest.hexdigest()
    predictions = pd.DataFrame({
        "row_id": np.flatnonzero(~fit_mask),
        "approval_month": raw_valid["RENT_APPROVAL_DATE"],
        "actual": y_valid,
    })
    results: dict[str, dict] = {}
    for name, spec in specs.items():
        metadata_path, predictions_path = directory / f"{name}.json", directory / f"{name}.npz"
        stored = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if stored.get("fingerprint") == fingerprint and stored.get("spec") == spec and predictions_path.exists():
            prediction = np.load(predictions_path, allow_pickle=False)["predictions"]
            if prediction.shape != (len(y_valid),):
                raise RuntimeError(f"Invalid cached prediction shape for {name}")
            results[name] = stored
            print(f"Cached {name}: RMSE={stored['metrics']['rmse']:.3f}", flush=True)
        else:
            print(f"Fitting {name} ...", flush=True)
            started = time.monotonic()
            prediction, best_iteration = fit_experiment(spec, x_fit, y_fit, x_valid, y_valid)
            stored = {
                "fingerprint": fingerprint, "spec": spec, "versions": versions,
                "best_iteration": best_iteration,
                "seconds": round(time.monotonic() - started, 3),
                "metrics": regression_metrics(y_valid.to_numpy(), prediction),
            }
            np.savez_compressed(predictions_path, predictions=prediction)
            save_json(metadata_path, stored)
            results[name] = stored
            print(f"Completed {name}: RMSE={stored['metrics']['rmse']:.3f}, MAE={stored['metrics']['mae']:.3f}, iterations={best_iteration}, {stored['seconds']:.1f}s", flush=True)
        predictions[name] = prediction

    payload = {
        "validation_start": args.validation_start,
        "fit_rows": len(raw_fit), "validation_rows": len(raw_valid),
        "feature_profile": "core", "feature_count": len(x_fit.columns),
        "evaluation_note": "Model-selection holdout; reused for early stopping and candidate/blend selection. Not an untouched test set.",
        "reference_iteration_note": "v2 controls use previously selected fixed iteration counts; all other candidates use early stopping or staged validation selection.",
        "experiments": results,
    }
    reference_names = list(reference["weights"])
    final_weights: dict[str, float] | None = None
    if set(reference_names).issubset(predictions.columns):
        baseline = predictions[reference_names].to_numpy() @ np.array([reference["weights"][name] for name in reference_names])
        predictions["v2_frozen_blend"] = baseline
        baseline_metrics = regression_metrics(y_valid.to_numpy(), baseline)
        best_metrics = baseline_metrics
        best_prediction = baseline
        winner, winner_weight = None, 0.0
        blend_results = {}
        for name in specs:
            if name in reference_names:
                continue
            alpha, blended = bounded_blend(y_valid.to_numpy(), baseline, predictions[name].to_numpy())
            metrics = regression_metrics(y_valid.to_numpy(), blended)
            blend_results[name] = {"candidate_weight": alpha, **metrics}
            if metrics["rmse"] < best_metrics["rmse"] - 0.05:
                best_metrics, best_prediction, winner, winner_weight = metrics, blended, name, alpha
        predictions["selected_blend"] = best_prediction
        final_weights = {
            name: float((1 - winner_weight) * reference["weights"][name])
            for name in reference_names if reference["weights"][name] > 1e-8
        }
        if winner:
            final_weights[winner] = winner_weight
        payload.update(
            baseline_metrics=baseline_metrics, blend_candidates=blend_results,
            selected_candidate=winner, selected_candidate_weight=winner_weight,
            selected_metrics=best_metrics, final_component_weights=final_weights,
            rmse_improvement=baseline_metrics["rmse"] - best_metrics["rmse"],
        )
        print(f"v2 reproduced: {baseline_metrics['rmse']:.6f}; selected: {best_metrics['rmse']:.6f}; added candidate: {winner} ({winner_weight:.3f})", flush=True)

    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    save_json(directory / "metrics.json", payload)
    print(f"Saved experiment evidence to {directory}", flush=True)

    if args.submission:
        if not final_weights:
            parser.error("Submission export requires all six v2 controls")
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Preparing full-data refit", flush=True)
        x_full, y_full, x_test = _prepare_partition(builder, CausalHistoryTransformer(24), raw, test, "core")
        test_prediction = np.zeros(len(test))
        for name, weight in final_weights.items():
            print(f"Refitting {name} (weight={weight:.4f}) ...", flush=True)
            count = max(50, int(results[name]["best_iteration"] * 1.05))
            prediction, _ = fit_experiment(specs[name], x_full, y_full, x_test, iterations=count)
            test_prediction += weight * prediction
        submission = make_submission(test_prediction)
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.submission, index=False)
        payload["submission"] = {
            "path": str(args.submission), "rows": len(submission),
            "sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
            "prediction_min": float(submission["Predicted"].min()),
            "prediction_max": float(submission["Predicted"].max()),
        }
        save_json(directory / "metrics.json", payload)
        print(f"Saved {len(submission):,} predictions to {args.submission}", flush=True)


if __name__ == "__main__":
    main()
