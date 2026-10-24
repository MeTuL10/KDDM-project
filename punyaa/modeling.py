"""LightGBM/CatBoost ensemble helpers for the Punyaa pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import CATEGORICAL_FEATURES


CATBOOST_SEEDS = (42, 2026, 5228)


@dataclass
class ValidationResult:
    predictions: np.ndarray
    component_predictions: np.ndarray
    component_names: list[str]
    weights: np.ndarray
    rmse: float
    mae: float
    best_iterations: dict[str, int]


def _lightgbm_frames(
    train: pd.DataFrame, *others: pd.DataFrame
) -> tuple[pd.DataFrame, ...]:
    prepared = [train.copy(), *(frame.copy() for frame in others)]
    for column in CATEGORICAL_FEATURES:
        categories = pd.Index(train[column].astype(str).unique())
        for frame in prepared:
            frame[column] = pd.Categorical(frame[column].astype(str), categories=categories)
    return tuple(prepared)


def _catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("unknown").astype(str)
    return result


def optimize_blend_weights(
    target: np.ndarray,
    predictions: np.ndarray,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 0.85,
    regularization: float = 0.002,
) -> np.ndarray:
    """Fit non-negative weights with mild shrinkage toward equal blending."""
    target = np.asarray(target, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] != len(target):
        raise ValueError("predictions must have shape (rows, models)")
    model_count = predictions.shape[1]
    equal = np.full(model_count, 1.0 / model_count)
    if lower_bound * model_count > 1 or upper_bound * model_count < 1:
        raise ValueError("blend bounds cannot satisfy a unit sum")
    target_scale = max(float(np.var(target)), 1.0)

    def objective(weights: np.ndarray) -> float:
        residual = target - predictions @ weights
        normalized_mse = float(np.mean(residual**2)) / target_scale
        penalty = regularization * float(np.sum((weights - equal) ** 2))
        return normalized_mse + penalty

    result = minimize(
        objective,
        equal,
        method="SLSQP",
        bounds=[(lower_bound, upper_bound)] * model_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success:
        return equal
    weights = np.clip(result.x, lower_bound, upper_bound)
    return weights / weights.sum()


def lightgbm_variant_parameters(quick: bool = False) -> dict[str, dict]:
    """Return deliberately different regularized LightGBM candidates."""
    iterations = 220 if quick else 2600
    learning_rate = 0.05 if quick else 0.02
    common = {
        "objective": "regression",
        "n_estimators": iterations,
        "learning_rate": learning_rate,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "max_bin": 511,
        "cat_smooth": 20.0,
        "cat_l2": 10.0,
        "verbosity": -1,
        "n_jobs": -1,
    }
    return {
        "regularized": {
            **common,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 80,
            "reg_alpha": 0.8,
            "reg_lambda": 7.0,
        },
        "balanced": {
            **common,
            "num_leaves": 63,
            "max_depth": 9,
            "min_child_samples": 60,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        },
        "detailed": {
            **common,
            "num_leaves": 127,
            "max_depth": 10,
            "min_child_samples": 90,
            "colsample_bytree": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 9.0,
        },
    }


def catboost_parameters(quick: bool = False) -> dict:
    return {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 240 if quick else 1600,
        "learning_rate": 0.05 if quick else 0.035,
        "depth": 8,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.7,
        "bagging_temperature": 0.5,
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": False,
    }


def fit_lightgbm_validation_variants(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    quick: bool = False,
    random_seed: int = 42,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Fit direct LightGBM variants and an optional median-benchmark residual."""
    import lightgbm as lgb

    x_train_lgb, x_valid_lgb = _lightgbm_frames(x_train, x_valid)
    names: list[str] = []
    predictions: list[np.ndarray] = []
    best_iterations: dict[str, int] = {}
    variants = lightgbm_variant_parameters(quick)

    for variant_name, parameters in variants.items():
        params = {**parameters, "random_state": random_seed}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            x_train_lgb,
            y_train,
            eval_X=x_valid_lgb,
            eval_y=y_valid,
            callbacks=[lgb.early_stopping(40 if quick else 120, verbose=False)],
            categorical_feature=CATEGORICAL_FEATURES,
        )
        component_name = f"lightgbm_{variant_name}"
        names.append(component_name)
        predictions.append(model.predict(x_valid_lgb))
        best_iterations[component_name] = int(
            model.best_iteration_ or params["n_estimators"]
        )

    benchmark_available = (
        "median_rent_benchmark" in x_train
        and "median_rent_benchmark" in x_valid
    )
    benchmark_train = (
        x_train["median_rent_benchmark"].to_numpy(dtype=float)
        if benchmark_available
        else np.array([np.nan])
    )
    benchmark_valid = (
        x_valid["median_rent_benchmark"].to_numpy(dtype=float)
        if benchmark_available
        else np.array([np.nan])
    )
    if np.isfinite(benchmark_train).all() and np.isfinite(benchmark_valid).all():
        params = {**variants["balanced"], "random_state": random_seed + 17}
        residual_model = lgb.LGBMRegressor(**params)
        residual_model.fit(
            x_train_lgb,
            y_train.to_numpy(dtype=float) - benchmark_train,
            eval_X=x_valid_lgb,
            eval_y=y_valid.to_numpy(dtype=float) - benchmark_valid,
            callbacks=[lgb.early_stopping(40 if quick else 120, verbose=False)],
            categorical_feature=CATEGORICAL_FEATURES,
        )
        component_name = "lightgbm_median_residual"
        names.append(component_name)
        predictions.append(benchmark_valid + residual_model.predict(x_valid_lgb))
        best_iterations[component_name] = int(
            residual_model.best_iteration_ or params["n_estimators"]
        )

    return names, np.column_stack(predictions), best_iterations


def fit_validation_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    quick: bool = False,
    random_seed: int = 42,
) -> ValidationResult:
    """Fit LightGBM variants and a multi-seed direct CatBoost ensemble."""
    from catboost import CatBoostRegressor

    names, lgb_predictions, best_iterations = fit_lightgbm_validation_variants(
        x_train,
        y_train,
        x_valid,
        y_valid,
        quick=quick,
        random_seed=random_seed,
    )
    component_predictions = [
        lgb_predictions[:, index] for index in range(lgb_predictions.shape[1])
    ]
    x_train_cb = _catboost_frame(x_train)
    x_valid_cb = _catboost_frame(x_valid)
    categorical_indices = [x_train_cb.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    seeds = (random_seed,) if quick else CATBOOST_SEEDS
    base_parameters = catboost_parameters(quick)

    for seed in seeds:
        model = CatBoostRegressor(**base_parameters, random_seed=seed)
        model.fit(
            x_train_cb,
            y_train,
            cat_features=categorical_indices,
            eval_set=(x_valid_cb, y_valid),
            early_stopping_rounds=40 if quick else 120,
            verbose=False,
        )
        component_name = f"catboost_seed_{seed}"
        names.append(component_name)
        component_predictions.append(model.predict(x_valid_cb))
        best_iterations[component_name] = max(1, int(model.get_best_iteration()) + 1)

    matrix = np.column_stack(component_predictions)
    weights = optimize_blend_weights(y_valid.to_numpy(dtype=float), matrix)
    predictions = matrix @ weights
    return ValidationResult(
        predictions=predictions,
        component_predictions=matrix,
        component_names=names,
        weights=weights,
        rmse=float(mean_squared_error(y_valid, predictions) ** 0.5),
        mae=float(mean_absolute_error(y_valid, predictions)),
        best_iterations=best_iterations,
    )


def fit_final_ensemble(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    component_names: list[str],
    weights: np.ndarray,
    best_iterations: dict[str, int],
    *,
    quick: bool = False,
    random_seed: int = 42,
) -> np.ndarray:
    """Refit selected components on all labelled data and predict test rows."""
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    weights = np.asarray(weights, dtype=float)
    if len(component_names) != len(weights):
        raise ValueError("component names and weights differ in length")
    final_prediction = np.zeros(len(x_test), dtype=float)
    x_train_lgb, x_test_lgb = _lightgbm_frames(x_train, x_test)
    x_train_cb = _catboost_frame(x_train)
    x_test_cb = _catboost_frame(x_test)
    categorical_indices = [x_train_cb.columns.get_loc(c) for c in CATEGORICAL_FEATURES]
    lgb_variants = lightgbm_variant_parameters(quick)
    cb_parameters = catboost_parameters(quick)

    for component_name, weight in zip(component_names, weights):
        if weight <= 1e-8:
            continue

        if component_name.startswith("lightgbm_"):
            variant = component_name.removeprefix("lightgbm_")
            is_residual = variant == "median_residual"
            parameter_key = "balanced" if is_residual else variant
            params = {
                **lgb_variants[parameter_key],
                "random_state": random_seed + (17 if is_residual else 0),
                "n_estimators": max(50, int(best_iterations[component_name] * 1.05)),
            }
            target = y_train.to_numpy(dtype=float)
            if is_residual:
                benchmark_train = x_train["median_rent_benchmark"].to_numpy(dtype=float)
                benchmark_test = x_test["median_rent_benchmark"].to_numpy(dtype=float)
                if not np.isfinite(benchmark_train).all() or not np.isfinite(
                    benchmark_test
                ).all():
                    raise ValueError("Median benchmark is incomplete")
                target = target - benchmark_train
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x_train_lgb,
                target,
                categorical_feature=CATEGORICAL_FEATURES,
            )
            prediction = model.predict(x_test_lgb)
            if is_residual:
                prediction = benchmark_test + prediction
        elif component_name.startswith("catboost_seed_"):
            seed = int(component_name.rsplit("_", 1)[1])
            params = {
                **cb_parameters,
                "random_seed": seed,
                "iterations": max(50, int(best_iterations[component_name] * 1.05)),
            }
            model = CatBoostRegressor(**params)
            model.fit(
                x_train_cb,
                y_train,
                cat_features=categorical_indices,
                verbose=False,
            )
            prediction = model.predict(x_test_cb)
        else:
            raise ValueError(f"Unknown component: {component_name}")
        final_prediction += float(weight) * prediction

    return final_prediction


def make_submission(predictions: np.ndarray) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("predictions must be a finite one-dimensional array")
    return pd.DataFrame(
        {"Id": np.arange(len(values), dtype=int), "Predicted": np.clip(values, 300, 9000)}
    )
