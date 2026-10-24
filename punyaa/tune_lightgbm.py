"""Evaluate LightGBM variants over rolling chronological validation windows."""

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
from .modeling import fit_lightgbm_validation_variants


ROLLING_WINDOWS = [
    ("2024-01", "2024-06"),
    ("2024-05", "2024-10"),
    ("2024-10", "2025-03"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/punyaa_lightgbm_rolling_metrics.json"),
    )
    parser.add_argument("--lag-months", type=int, default=24)
    parser.add_argument(
        "--feature-profile", choices=["core", "recent"], default="core"
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.data_dir / "train.csv")
    dates = pd.to_datetime(raw["RENT_APPROVAL_DATE"], format="%Y-%m")
    builder = StaticFeatureBuilder(args.data_dir)
    static = builder.transform(raw.drop(columns=[TARGET]))
    target = raw[TARGET].astype(float).reset_index(drop=True)
    results: list[dict] = []

    for validation_start_text, validation_end_text in ROLLING_WINDOWS:
        validation_start = pd.Period(validation_start_text, freq="M").to_timestamp()
        validation_end = pd.Period(validation_end_text, freq="M").to_timestamp()
        train_mask = dates < validation_start
        valid_mask = (dates >= validation_start) & (dates <= validation_end)
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Invalid rolling window {validation_start_text}")

        fit_static = static.loc[train_mask].reset_index(drop=True)
        valid_static = static.loc[valid_mask].reset_index(drop=True)
        y_fit = target.loc[train_mask].reset_index(drop=True)
        y_valid = target.loc[valid_mask].reset_index(drop=True)
        history = CausalHistoryTransformer(args.lag_months)
        fit_history = history.fit_transform_train(fit_static, y_fit)
        valid_history = history.transform(valid_static)
        x_fit = assemble_model_features(
            fit_static, fit_history, feature_profile=args.feature_profile
        )
        x_valid = assemble_model_features(
            valid_static, valid_history, feature_profile=args.feature_profile
        )

        names, predictions, iterations = fit_lightgbm_validation_variants(
            x_fit,
            y_fit,
            x_valid,
            y_valid,
            quick=args.quick,
        )
        for index, name in enumerate(names):
            rmse = float(
                np.mean((y_valid.to_numpy() - predictions[:, index]) ** 2) ** 0.5
            )
            row = {
                "validation_start": validation_start_text,
                "validation_end": validation_end_text,
                "fit_rows": int(train_mask.sum()),
                "validation_rows": int(valid_mask.sum()),
                "component": name,
                "rmse": rmse,
                "best_iteration": iterations[name],
            }
            results.append(row)
            print(
                f"{validation_start_text}..{validation_end_text} | "
                f"{name}: RMSE={rmse:.3f}"
            )

    result_frame = pd.DataFrame(results)
    summary = (
        result_frame.groupby("component")["rmse"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean")
    )
    payload = {
        "quick": args.quick,
        "lag_months": args.lag_months,
        "feature_profile": args.feature_profile,
        "windows": results,
        "summary": {
            component: {metric: float(value) for metric, value in row.items()}
            for component, row in summary.to_dict(orient="index").items()
        },
        "selected_component": str(summary.index[0]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nRolling summary:")
    print(summary.round(3).to_string())
    print(f"Saved rolling metrics to {args.output}")


if __name__ == "__main__":
    main()
