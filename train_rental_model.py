#!/usr/bin/env python3
"""Train a leakage-safe HDB rental model and create a Kaggle submission.

Example
-------
python train_rental_model.py --data-dir data --output data/submission_model.csv

Install the only non-standard dependency first if necessary:
    pip install catboost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "MONTHLY_RENT"
EARTH_RADIUS_KM = 6371.0088


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def haversine_nearest(
    latitude: pd.Series,
    longitude: pd.Series,
    points: pd.DataFrame,
    chunk_size: int = 4000,
) -> np.ndarray:
    """Return distance in kilometres to the nearest point of interest."""
    result = np.full(len(latitude), np.nan, dtype=np.float32)
    valid_points = points[["LATITUDE", "LONGITUDE"]].dropna().drop_duplicates()
    if valid_points.empty:
        return result

    point_lat = np.radians(valid_points["LATITUDE"].to_numpy(dtype=float))[None, :]
    point_lon = np.radians(valid_points["LONGITUDE"].to_numpy(dtype=float))[None, :]
    lat_values = latitude.to_numpy(dtype=float)
    lon_values = longitude.to_numpy(dtype=float)

    for start in range(0, len(latitude), chunk_size):
        stop = min(start + chunk_size, len(latitude))
        lat = lat_values[start:stop]
        lon = lon_values[start:stop]
        valid = np.isfinite(lat) & np.isfinite(lon)
        if not valid.any():
            continue

        lat_rad = np.radians(lat[valid])[:, None]
        lon_rad = np.radians(lon[valid])[:, None]
        dlat = point_lat - lat_rad
        dlon = point_lon - lon_rad
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat_rad) * np.cos(point_lat) * np.sin(dlon / 2.0) ** 2
        )
        distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        chunk_result = np.full(stop - start, np.nan, dtype=np.float32)
        chunk_result[valid] = distance.min(axis=1).astype(np.float32)
        result[start:stop] = chunk_result
    return result


def add_location_features(frame: pd.DataFrame, auxiliary_dir: Path) -> pd.DataFrame:
    blocks = pd.read_csv(auxiliary_dir / "sg-hdb-block.csv")
    for column in ["TOWN", "BLOCK", "STREET"]:
        blocks[column] = clean_text(blocks[column])
    blocks = blocks.drop_duplicates(["TOWN", "BLOCK", "STREET"])

    block_columns = [
        "TOWN",
        "BLOCK",
        "STREET",
        "POSTAL_CODE",
        "LATITUDE",
        "LONGITUDE",
        "MAX_FLOOR",
        "YEAR_COMPLETED",
        "SUBZONE",
        "PLANNING_AREA",
        "REGION",
    ]
    frame = frame.merge(blocks[block_columns], on=["TOWN", "BLOCK", "STREET"], how="left")

    mrt = pd.read_csv(auxiliary_dir / "sg-mrt-stations.csv")
    schools = pd.read_csv(auxiliary_dir / "sg-schools.csv")
    malls = pd.read_csv(auxiliary_dir / "sg-shopping-malls.csv")
    frame["DIST_MRT_KM"] = haversine_nearest(frame["LATITUDE"], frame["LONGITUDE"], mrt)
    frame["DIST_SCHOOL_KM"] = haversine_nearest(
        frame["LATITUDE"], frame["LONGITUDE"], schools
    )
    frame["DIST_MALL_KM"] = haversine_nearest(
        frame["LATITUDE"], frame["LONGITUDE"], malls
    )
    return frame


def add_economic_features(frame: pd.DataFrame, auxiliary_dir: Path) -> pd.DataFrame:
    coe = pd.read_csv(auxiliary_dir / "sg-coe-prices.csv")
    month_number = {
        month: number
        for number, month in enumerate(
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
            start=1,
        )
    }
    coe["MONTH_NUMBER"] = clean_text(coe["MONTH"]).map(month_number)
    coe["DATE_KEY"] = (
        coe["YEAR"].astype(int).astype(str)
        + "-"
        + coe["MONTH_NUMBER"].astype(int).astype(str).str.zfill(2)
    )
    for column in ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM"]:
        coe[column] = pd.to_numeric(
            coe[column].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )
    coe_monthly = (
        coe.groupby("DATE_KEY")
        .agg(
            COE_PREMIUM=("QUOTA_PREMIUM", "mean"),
            COE_PREVAILING_PREMIUM=("PREVAILING_QUOTA_PREMIUM", "mean"),
        )
        .reset_index()
    )
    return frame.merge(coe_monthly, on="DATE_KEY", how="left")


def build_base_features(all_rows: pd.DataFrame, auxiliary_dir: Path) -> pd.DataFrame:
    frame = all_rows.copy()
    text_columns = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED"]
    for column in text_columns:
        frame[column] = clean_text(frame[column])
    frame["FLAT_TYPE"] = frame["FLAT_TYPE"].str.replace(" room", "-room", regex=False)

    date = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m")
    frame["DATE_KEY"] = date.dt.strftime("%Y-%m")
    frame["YEAR"] = date.dt.year
    frame["MONTH"] = date.dt.month
    frame["QUARTER"] = date.dt.quarter
    frame["MONTH_INDEX"] = (date.dt.year - 2021) * 12 + date.dt.month - 1
    frame["MONTH_SIN"] = np.sin(2 * np.pi * frame["MONTH"] / 12)
    frame["MONTH_COS"] = np.cos(2 * np.pi * frame["MONTH"] / 12)
    frame["FLAT_AGE"] = frame["YEAR"] - frame["LEASE_COMMENCE_DATE"]

    frame["PROPERTY"] = frame["TOWN"] + "|" + frame["BLOCK"] + "|" + frame["STREET"]
    frame["PROPERTY_FLAT_TYPE"] = frame["PROPERTY"] + "|" + frame["FLAT_TYPE"]
    frame["STREET_FLAT_TYPE"] = frame["TOWN"] + "|" + frame["STREET"] + "|" + frame["FLAT_TYPE"]
    frame["TOWN_FLAT_TYPE"] = frame["TOWN"] + "|" + frame["FLAT_TYPE"]
    frame["MODEL_FLAT_TYPE"] = frame["FLAT_MODEL"] + "|" + frame["FLAT_TYPE"]

    frame = add_location_features(frame, auxiliary_dir)
    frame = add_economic_features(frame, auxiliary_dir)

    frame = frame.drop(columns=["RENT_APPROVAL_DATE", "DATE_KEY"])
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].fillna("unknown").astype(str)
    return frame


HISTORY_SPECS = [
    ("PROPERTY_FLAT_TYPE", 6.0),
    ("PROPERTY", 8.0),
    ("STREET_FLAT_TYPE", 15.0),
    ("TOWN_FLAT_TYPE", 30.0),
    ("MODEL_FLAT_TYPE", 30.0),
]


def add_causal_target_history(
    reference_x: pd.DataFrame, reference_y: pd.Series
) -> pd.DataFrame:
    """Build training encodings using strictly earlier calendar months."""
    output = reference_x.copy()
    history = reference_x.copy()
    history["_target"] = reference_y.to_numpy(dtype=float)
    first_month = int(history["MONTH_INDEX"].min())
    last_month = int(history["MONTH_INDEX"].max())
    months = np.arange(first_month, last_month + 1)

    global_monthly = history.groupby("MONTH_INDEX")["_target"].agg(["sum", "count"])
    global_monthly = global_monthly.reindex(months, fill_value=0)
    global_sum_before = global_monthly["sum"].cumsum().shift(1)
    global_count_before = global_monthly["count"].cumsum().shift(1)
    global_prior = global_sum_before / global_count_before
    # The first observed month has no legitimate historical target information.
    # Keep its prior as NaN rather than filling it with a mean computed from current
    # or future labels. CatBoost handles missing numerical values natively.
    row_global_prior = output["MONTH_INDEX"].map(global_prior)
    output["PRIOR_GLOBAL_MEAN"] = row_global_prior.to_numpy()

    for column, smoothing in HISTORY_SPECS:
        monthly = (
            history.groupby([column, "MONTH_INDEX"], observed=True)["_target"]
            .agg(["sum", "count"])
            .reset_index()
        )
        groups = history[column].drop_duplicates().to_numpy()
        grid = pd.MultiIndex.from_product(
            [groups, months], names=[column, "MONTH_INDEX"]
        ).to_frame(index=False)
        grid = grid.merge(monthly, on=[column, "MONTH_INDEX"], how="left")
        grid[["sum", "count"]] = grid[["sum", "count"]].fillna(0)
        grouped = grid.groupby(column, sort=False, observed=True)
        grid["all_sum"] = grouped["sum"].cumsum() - grid["sum"]
        grid["all_count"] = grouped["count"].cumsum() - grid["count"]

        lookup = grid.set_index([column, "MONTH_INDEX"])
        row_keys = pd.MultiIndex.from_frame(output[[column, "MONTH_INDEX"]])
        sums = lookup["all_sum"].reindex(row_keys).to_numpy(dtype=float)
        counts = lookup["all_count"].reindex(row_keys).to_numpy(dtype=float)
        prior = row_global_prior.to_numpy(dtype=float)
        output[f"HIST_MEAN_{column}"] = (sums + smoothing * prior) / (
            counts + smoothing
        )
        output[f"HIST_COUNT_{column}"] = counts

        if column in {"PROPERTY_FLAT_TYPE", "STREET_FLAT_TYPE", "TOWN_FLAT_TYPE"}:
            for window in (6, 12, 24):
                grid[f"recent_sum_{window}"] = grouped["sum"].transform(
                    lambda values: values.shift(1).rolling(window, min_periods=1).sum()
                )
                grid[f"recent_count_{window}"] = grouped["count"].transform(
                    lambda values: values.shift(1).rolling(window, min_periods=1).sum()
                )
                lookup = grid.set_index([column, "MONTH_INDEX"])
                recent_sum = (
                    lookup[f"recent_sum_{window}"]
                    .reindex(row_keys)
                    .fillna(0)
                    .to_numpy(dtype=float)
                )
                recent_count = (
                    lookup[f"recent_count_{window}"]
                    .reindex(row_keys)
                    .fillna(0)
                    .to_numpy(dtype=float)
                )
                output[f"RECENT_{window}_MEAN_{column}"] = (
                    recent_sum + smoothing * prior
                ) / (recent_count + smoothing)
                output[f"RECENT_{window}_COUNT_{column}"] = recent_count
    return output


def add_reference_target_history(
    reference_x: pd.DataFrame,
    reference_y: pd.Series,
    target_x: pd.DataFrame,
) -> pd.DataFrame:
    """Map statistics from labelled reference months onto future rows."""
    output = target_x.copy()
    history = reference_x.copy()
    history["_target"] = reference_y.to_numpy(dtype=float)
    global_mean = float(reference_y.mean())
    last_month = int(reference_x["MONTH_INDEX"].max())
    output["PRIOR_GLOBAL_MEAN"] = global_mean

    for column, smoothing in HISTORY_SPECS:
        stats = history.groupby(column, observed=True)["_target"].agg(["sum", "count"])
        sums = output[column].map(stats["sum"]).fillna(0).astype(float)
        counts = output[column].map(stats["count"]).fillna(0).astype(float)
        output[f"HIST_MEAN_{column}"] = (sums + smoothing * global_mean) / (
            counts + smoothing
        )
        output[f"HIST_COUNT_{column}"] = counts

        if column in {"PROPERTY_FLAT_TYPE", "STREET_FLAT_TYPE", "TOWN_FLAT_TYPE"}:
            for window in (6, 12, 24):
                recent = history[
                    history["MONTH_INDEX"] >= last_month - window + 1
                ]
                recent_global = float(recent["_target"].mean())
                recent_stats = recent.groupby(column, observed=True)["_target"].agg(
                    ["sum", "count"]
                )
                recent_sum = output[column].map(recent_stats["sum"]).fillna(0).astype(float)
                recent_count = (
                    output[column].map(recent_stats["count"]).fillna(0).astype(float)
                )
                output[f"RECENT_{window}_MEAN_{column}"] = (
                    recent_sum + smoothing * recent_global
                ) / (recent_count + smoothing)
                output[f"RECENT_{window}_COUNT_{column}"] = recent_count
    return output


def prepare_for_catboost(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    categorical = output.select_dtypes(include="object").columns.tolist()
    for column in categorical:
        output[column] = output[column].fillna("unknown").astype(str)
    numeric = output.columns.difference(categorical)
    output[numeric] = output[numeric].apply(pd.to_numeric, errors="coerce")
    return output, categorical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/submission_model.csv"))
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Stop after time validation without fitting final models.",
    )
    args = parser.parse_args()

    try:
        from catboost import CatBoostRegressor
    except ImportError as error:
        raise SystemExit(
            "CatBoost is required. Install it with: pip install catboost"
        ) from error

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    auxiliary_dir = args.data_dir / "auxiliary"
    if TARGET not in train:
        raise ValueError(f"{TARGET} is missing from train.csv")

    train_dates = pd.to_datetime(train["RENT_APPROVAL_DATE"], format="%Y-%m")
    combined = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    combined_features = build_base_features(combined, auxiliary_dir)
    x_train_base = combined_features.iloc[: len(train)].reset_index(drop=True)
    x_test_base = combined_features.iloc[len(train) :].reset_index(drop=True)
    y = train[TARGET].astype(float).reset_index(drop=True)

    # Hold out the most recent months so validation resembles forecasting.
    last_month = train_dates.max().to_period("M")
    validation_start = (last_month - args.validation_months + 1).to_timestamp()
    fit_mask = (train_dates < validation_start).to_numpy()
    valid_mask = ~fit_mask

    x_fit_base = x_train_base.loc[fit_mask].reset_index(drop=True)
    y_fit = y.loc[fit_mask].reset_index(drop=True)
    x_valid_base = x_train_base.loc[valid_mask].reset_index(drop=True)
    y_valid = y.loc[valid_mask].reset_index(drop=True)

    x_fit = add_causal_target_history(x_fit_base, y_fit)
    x_valid = add_reference_target_history(x_fit_base, y_fit, x_valid_base)
    x_fit, categorical = prepare_for_catboost(x_fit)
    x_valid, _ = prepare_for_catboost(x_valid)

    validation_predictions = np.zeros(len(x_valid), dtype=float)
    best_iterations: list[int] = []
    for seed in args.seeds:
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=args.iterations,
            learning_rate=0.045,
            depth=9,
            l2_leaf_reg=8.0,
            random_seed=seed,
            random_strength=0.4,
            bagging_temperature=0.5,
            verbose=100,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(
            x_fit,
            y_fit,
            cat_features=categorical,
            eval_set=(x_valid, y_valid),
            early_stopping_rounds=150,
            verbose=100,
        )
        validation_predictions += model.predict(x_valid) / len(args.seeds)
        best_iterations.append(max(200, model.get_best_iteration() + 1))

    validation_rmse = float(
        np.sqrt(np.mean((y_valid.to_numpy() - validation_predictions) ** 2))
    )
    print(
        f"Time validation: {validation_start:%Y-%m} to {train_dates.max():%Y-%m}; "
        f"RMSE={validation_rmse:.3f}"
    )
    diagnostics = pd.DataFrame(
        {
            "month": train_dates.loc[valid_mask].dt.strftime("%Y-%m").to_numpy(),
            "actual": y_valid.to_numpy(),
            "predicted": validation_predictions,
        }
    )
    monthly_diagnostics = diagnostics.groupby("month").agg(
        actual=("actual", "mean"), predicted=("predicted", "mean")
    )
    monthly_diagnostics["bias"] = (
        monthly_diagnostics["predicted"] - monthly_diagnostics["actual"]
    )
    print("Validation monthly means and prediction bias:")
    print(monthly_diagnostics.round(2).to_string())
    if args.validation_only:
        return

    # Rebuild history features using every labelled row, then refit each seed.
    x_full = add_causal_target_history(x_train_base, y)
    x_test = add_reference_target_history(x_train_base, y, x_test_base)
    x_full, categorical = prepare_for_catboost(x_full)
    x_test, _ = prepare_for_catboost(x_test)

    test_predictions = np.zeros(len(test), dtype=float)
    for seed, best_iteration in zip(args.seeds, best_iterations):
        final_iterations = min(args.iterations, int(best_iteration * 1.10))
        model = CatBoostRegressor(
            loss_function="RMSE",
            iterations=final_iterations,
            learning_rate=0.045,
            depth=9,
            l2_leaf_reg=8.0,
            random_seed=seed,
            random_strength=0.4,
            bagging_temperature=0.5,
            verbose=100,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x_full, y, cat_features=categorical, verbose=100)
        test_predictions += model.predict(x_test) / len(args.seeds)

    test_predictions = np.clip(test_predictions, 300, 9000)
    submission = pd.DataFrame(
        {"Id": np.arange(len(test), dtype=int), "Predicted": test_predictions}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
