"""Train and validate Punyaa's leakage-safe HDB rent ensemble.

Examples
--------
Run a fast end-to-end validation smoke test::

    python -m punyaa.train --quick --validation-only

Run the production configuration and create a Kaggle submission::

    python -m punyaa.train --output outputs/punyaa_submission.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (
    TARGET,
    CausalHistoryTransformer,
    StaticFeatureBuilder,
    assemble_model_features,
)
from .modeling import fit_final_ensemble, fit_validation_ensemble, make_submission


def _prepare_partition(
    builder: StaticFeatureBuilder,
    history: CausalHistoryTransformer,
    raw_train: pd.DataFrame,
    raw_future: pd.DataFrame,
    feature_profile: str = "core",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    y_train = raw_train[TARGET].astype(float).reset_index(drop=True)
    train_static = builder.transform(raw_train.drop(columns=[TARGET]))
    future_static = builder.transform(raw_future.drop(columns=[TARGET], errors="ignore"))
    train_history = history.fit_transform_train(train_static, y_train)
    future_history = history.transform(future_static)
    return (
        assemble_model_features(
            train_static, train_history, feature_profile=feature_profile
        ),
        y_train,
        assemble_model_features(
            future_static, future_history, feature_profile=feature_profile
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/punyaa_submission.csv"))
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/punyaa_validation_metrics.json"),
    )
    parser.add_argument("--validation-start", default="2024-05")
    parser.add_argument("--lag-months", type=int, default=24)
    parser.add_argument(
        "--feature-profile",
        choices=["core", "recent"],
        default="core",
        help="Use proven core histories or experimental 6/12/24-month windows",
    )
    parser.add_argument("--quick", action="store_true", help="Use small models for a smoke run")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    if TARGET not in train:
        raise ValueError(f"{TARGET} is missing from train.csv")

    dates = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m")
    validation_start = pd.Period(args.validation_start, freq="M").to_timestamp()
    fit_mask = dates < validation_start
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("validation-start must split the labelled data")

    raw_fit = train.loc[fit_mask].reset_index(drop=True)
    raw_valid = train.loc[~fit_mask].reset_index(drop=True)
    builder = StaticFeatureBuilder(args.data_dir)
    history = CausalHistoryTransformer(lag_months=args.lag_months)
    x_fit, y_fit, x_valid = _prepare_partition(
        builder, history, raw_fit, raw_valid, args.feature_profile
    )
    y_valid = raw_valid[TARGET].astype(float).reset_index(drop=True)

    print(
        f"Validation split: {len(raw_fit):,} rows before {args.validation_start}; "
        f"{len(raw_valid):,} rows from {args.validation_start} onward"
    )
    result = fit_validation_ensemble(
        x_fit, y_fit, x_valid, y_valid, quick=args.quick
    )
    component_rmse = {
        name: float(np.mean((y_valid.to_numpy() - result.component_predictions[:, index]) ** 2) ** 0.5)
        for index, name in enumerate(result.component_names)
    }
    metrics = {
        "validation_start": args.validation_start,
        "validation_rows": len(raw_valid),
        "quick": args.quick,
        "lag_months": args.lag_months,
        "feature_profile": args.feature_profile,
        "rmse": result.rmse,
        "mae": result.mae,
        "component_rmse": component_rmse,
        "weights": dict(zip(result.component_names, map(float, result.weights))),
        "best_iterations": result.best_iterations,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved validation metrics to {args.metrics_output}")

    if args.validation_only:
        return

    final_history = CausalHistoryTransformer(lag_months=args.lag_months)
    x_full, y_full, x_test = _prepare_partition(
        builder,
        final_history,
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        args.feature_profile,
    )
    predictions = fit_final_ensemble(
        x_full,
        y_full,
        x_test,
        result.component_names,
        result.weights,
        result.best_iterations,
        quick=args.quick,
    )
    submission = make_submission(predictions)
    if len(submission) != len(test):
        raise RuntimeError("Submission row count does not match test.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
