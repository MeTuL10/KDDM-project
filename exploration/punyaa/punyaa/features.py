"""Leakage-safe feature construction for HDB monthly-rent prediction.

The implementation combines the strongest repeatable ideas documented in the
repository: a compact static feature set, native categorical values, named
geographic context, an exact 24-month market anchor, and smoothed historical
rent levels that exclude the current and future calendar months.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


TARGET = "MONTHLY_RENT"
LATITUDE_KM = 111.0
LONGITUDE_KM = 110.96

BASE_FEATURES = [
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "FLOOR_AREA_SQM",
    "approval_date_num",
    "approval_month",
    "coe_quota_premium_mean",
    "YEAR_COMPLETED",
    "MAX_FLOOR",
    "mrt_dist_nearest_km",
    "mall_dist_nearest_km",
    "nearest_mrt_name",
    "nearest_tourist_attraction_name",
    "nearest_monument_name",
    "median_rent_benchmark",
]

HISTORY_WINDOWS = (6, 12, 24)

GROUP_SPECS = {
    "town_flat_type": (["TOWN", "FLAT_TYPE"], 30.0),
    "property_flat_type": (["TOWN", "BLOCK", "STREET", "FLAT_TYPE"], 8.0),
}

CORE_HISTORY_FEATURES = [
    "monthly_target_lag",
    "town_flat_type_history_mean",
    "town_flat_type_history_count",
    "property_flat_type_history_mean",
    "property_flat_type_history_count",
]

RECENT_HISTORY_FEATURES = [
    *(f"global_recent_{window}_mean" for window in HISTORY_WINDOWS),
    "global_recent_6_24_change",
    *(
        feature
        for group in GROUP_SPECS
        for window in HISTORY_WINDOWS
        for feature in (
            f"{group}_recent_{window}_mean",
            f"{group}_recent_{window}_count",
        )
    ),
    *(f"{group}_recent_6_24_change" for group in GROUP_SPECS),
]

HISTORY_FEATURES = CORE_HISTORY_FEATURES + RECENT_HISTORY_FEATURES

MODEL_FEATURE_PROFILES = {
    "core": CORE_HISTORY_FEATURES,
    "recent": HISTORY_FEATURES,
}

MODEL_FEATURES = BASE_FEATURES + HISTORY_FEATURES

CATEGORICAL_FEATURES = [
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "nearest_mrt_name",
    "nearest_tourist_attraction_name",
    "nearest_monument_name",
]

def normalize_category(series: pd.Series) -> pd.Series:
    """Return the repository's canonical representation of a category."""
    result = series.astype("string").str.lower().str.strip()
    result = result.str.replace(r"[-_/]+", " ", regex=True)
    result = result.str.replace(r"\s+", " ", regex=True).str.strip()
    return result.fillna("unknown")


def _nearest(
    row_latitude: pd.Series,
    row_longitude: pd.Series,
    points: pd.DataFrame,
    *,
    name_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Find the nearest point and optionally return its name."""
    valid = points.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)
    point_coordinates = valid[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float, copy=True)
    point_coordinates *= [LATITUDE_KM, LONGITUDE_KM]
    row_coordinates = np.column_stack(
        [
            row_latitude.to_numpy(dtype=float) * LATITUDE_KM,
            row_longitude.to_numpy(dtype=float) * LONGITUDE_KM,
        ]
    )
    distances, indices = cKDTree(point_coordinates).query(row_coordinates, k=1)
    names = None
    if name_column is not None:
        names = valid[name_column].astype(str).to_numpy()[indices]
    return distances.astype(np.float32), names


def _read_geojson_points(path: Path, name_field: str) -> pd.DataFrame:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    records = []
    for feature in document.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        name = (feature.get("properties") or {}).get(name_field)
        if name is None:
            continue
        records.append(
            {"LONGITUDE": coordinates[0], "LATITUDE": coordinates[1], "NAME": name}
        )
    if not records:
        raise ValueError(f"No named Point features found in {path}")
    return pd.DataFrame(records)


class StaticFeatureBuilder:
    """Build non-target features from the supplied project datasets."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        auxiliary = self.data_dir / "auxiliary"

        hdb = pd.read_csv(auxiliary / "sg-hdb-block.csv")
        for column in ["TOWN", "BLOCK", "STREET"]:
            hdb[column] = normalize_category(hdb[column])
        self.hdb = hdb.drop_duplicates(["TOWN", "BLOCK", "STREET"])
        self.hdb_town_medians = self.hdb.groupby("TOWN")[[
            "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"
        ]].median()
        self.hdb_global_medians = self.hdb[[
            "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"
        ]].median()

        self.mrt = pd.read_csv(auxiliary / "sg-mrt-stations.csv")
        self.malls = pd.read_csv(auxiliary / "sg-shopping-malls.csv")
        self.coe = self._prepare_coe(pd.read_csv(auxiliary / "sg-coe-prices.csv"))
        median_path = auxiliary / "sg-hdb-median-rent.csv"
        self.median_rent = (
            self._prepare_median_rent(pd.read_csv(median_path))
            if median_path.exists()
            else None
        )

        extra = self.data_dir / "possible_additional_features"
        self.attractions = _read_geojson_points(
            extra / "TouristAttractions.geojson", "PAGETITLE"
        )
        self.monuments = _read_geojson_points(
            extra / "MonumentsGEOJSON.geojson", "NAME"
        )

    @staticmethod
    def _prepare_coe(coe: pd.DataFrame) -> pd.DataFrame:
        month_numbers = {
            name: number
            for number, name in enumerate(
                [
                    "jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec",
                ],
                start=1,
            )
        }
        result = coe.copy()
        result["approval_month"] = normalize_category(result["MONTH"]).map(month_numbers)
        result["approval_year"] = result["YEAR"].astype(int)
        result["QUOTA_PREMIUM"] = pd.to_numeric(
            result["QUOTA_PREMIUM"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
        return (
            result.groupby(["approval_year", "approval_month"], as_index=False)
            .agg(coe_quota_premium_mean=("QUOTA_PREMIUM", "mean"))
        )

    @staticmethod
    def _prepare_median_rent(raw: pd.DataFrame) -> pd.DataFrame:
        """Prepare the optional official quarterly HDB rental benchmark."""
        result = raw.copy()
        result["TOWN"] = normalize_category(result["town"])
        flat_type_map = {
            "1-RM": "1 room",
            "2-RM": "2 room",
            "3-RM": "3 room",
            "4-RM": "4 room",
            "5-RM": "5 room",
            "EXEC": "executive",
        }
        result["FLAT_TYPE"] = result["flat_type"].map(flat_type_map)
        result["median_rent_benchmark"] = pd.to_numeric(
            result["median_rent"], errors="coerce"
        )
        result["quarter_key"] = result["quarter"].astype(str)

        national = (
            result.groupby(["quarter_key", "FLAT_TYPE"])["median_rent_benchmark"]
            .median()
            .rename("national_median")
            .reset_index()
        )
        result = result.merge(national, on=["quarter_key", "FLAT_TYPE"], how="left")
        result["median_rent_benchmark"] = result["median_rent_benchmark"].fillna(
            result["national_median"]
        )

        # The supplied series ends at 2026-Q2 while the test set includes July
        # 2026. Extend one quarter using a clipped Q1-to-Q2 change.
        q1 = result[result["quarter_key"].eq("2026-Q1")].set_index(
            ["TOWN", "FLAT_TYPE"]
        )["median_rent_benchmark"]
        q2 = result[result["quarter_key"].eq("2026-Q2")].copy()
        if not q2.empty:
            keys = q2.set_index(["TOWN", "FLAT_TYPE"]).index
            previous = q1.reindex(keys).to_numpy(dtype=float)
            current = q2["median_rent_benchmark"].to_numpy(dtype=float)
            change = np.nan_to_num(current - previous, nan=25.0)
            q3 = q2.copy()
            q3["quarter_key"] = "2026-Q3"
            q3["median_rent_benchmark"] = current + np.clip(change, 0.0, 60.0)
            result = pd.concat([result, q3], ignore_index=True)

        return result[[
            "quarter_key", "TOWN", "FLAT_TYPE", "median_rent_benchmark"
        ]].drop_duplicates(["quarter_key", "TOWN", "FLAT_TYPE"])

    def _fill_hdb_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]:
            result[column] = result[column].fillna(
                result["TOWN"].map(self.hdb_town_medians[column])
            )
            result[column] = result[column].fillna(self.hdb_global_medians[column])
        return result

    def transform(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        """Create static features without reading any target values."""
        frame = raw_frame.copy().reset_index(drop=True)
        for column in ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]:
            frame[column] = normalize_category(frame[column])

        dates = pd.to_datetime(frame["RENT_APPROVAL_DATE"], format="%Y-%m")
        frame["approval_year"] = dates.dt.year.astype(int)
        frame["approval_month"] = dates.dt.month.astype(int)
        frame["approval_date_num"] = frame["approval_year"] * 12 + frame["approval_month"]
        frame["MONTH_INDEX"] = frame["approval_date_num"]
        frame["quarter_key"] = (
            frame["approval_year"].astype(str)
            + "-Q"
            + (((frame["approval_month"] - 1) // 3) + 1).astype(str)
        )

        hdb_columns = [
            "TOWN", "BLOCK", "STREET", "LATITUDE", "LONGITUDE",
            "MAX_FLOOR", "YEAR_COMPLETED",
        ]
        frame = frame.merge(
            self.hdb[hdb_columns], on=["TOWN", "BLOCK", "STREET"], how="left"
        )
        frame = self._fill_hdb_values(frame)

        frame["mrt_dist_nearest_km"], mrt_names = _nearest(
            frame["LATITUDE"], frame["LONGITUDE"], self.mrt, name_column="NAME"
        )
        frame["nearest_mrt_name"] = mrt_names
        frame["mall_dist_nearest_km"], _ = _nearest(
            frame["LATITUDE"], frame["LONGITUDE"], self.malls
        )
        _, attraction_names = _nearest(
            frame["LATITUDE"], frame["LONGITUDE"], self.attractions, name_column="NAME"
        )
        frame["nearest_tourist_attraction_name"] = attraction_names
        _, monument_names = _nearest(
            frame["LATITUDE"], frame["LONGITUDE"], self.monuments, name_column="NAME"
        )
        frame["nearest_monument_name"] = monument_names

        frame = frame.merge(
            self.coe, on=["approval_year", "approval_month"], how="left"
        )
        if frame["coe_quota_premium_mean"].isna().any():
            monthly = self.coe.assign(
                month_key=lambda x: x["approval_year"] * 12 + x["approval_month"]
            ).set_index("month_key")["coe_quota_premium_mean"].sort_index()
            known_months = monthly.index.to_numpy(dtype=float)
            known_values = monthly.to_numpy(dtype=float)
            missing = frame["coe_quota_premium_mean"].isna()
            frame.loc[missing, "coe_quota_premium_mean"] = np.interp(
                frame.loc[missing, "MONTH_INDEX"].to_numpy(dtype=float),
                known_months,
                known_values,
            )

        if self.median_rent is None:
            frame["median_rent_benchmark"] = np.nan
        else:
            frame = frame.merge(
                self.median_rent,
                on=["quarter_key", "TOWN", "FLAT_TYPE"],
                how="left",
            )
            quarter_type_median = self.median_rent.groupby(
                ["quarter_key", "FLAT_TYPE"]
            )["median_rent_benchmark"].median()
            keys = pd.MultiIndex.from_frame(frame[["quarter_key", "FLAT_TYPE"]])
            fallback = pd.Series(keys.map(quarter_type_median), index=frame.index)
            frame["median_rent_benchmark"] = frame["median_rent_benchmark"].fillna(
                fallback
            )

        for column in CATEGORICAL_FEATURES:
            frame[column] = normalize_category(frame[column])
        return frame


def _group_key(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    return frame[columns].astype(str).agg("\x1f".join, axis=1)


class CausalHistoryTransformer:
    """Create target features using labels from strictly earlier months."""

    def __init__(self, lag_months: int = 24):
        if lag_months < 1:
            raise ValueError("lag_months must be positive")
        self.lag_months = int(lag_months)
        self._is_fitted = False

    def fit(self, reference: pd.DataFrame, target: pd.Series) -> "CausalHistoryTransformer":
        if len(reference) != len(target):
            raise ValueError("reference and target lengths differ")
        history = reference[["MONTH_INDEX"]].copy().reset_index(drop=True)
        for name, (columns, _) in GROUP_SPECS.items():
            history[f"__key_{name}"] = _group_key(reference.reset_index(drop=True), columns)
        history["__target"] = pd.Series(target).reset_index(drop=True).astype(float)
        self.history_ = history
        self.monthly_mean_ = history.groupby("MONTH_INDEX")["__target"].mean()
        self._is_fitted = True
        return self

    def fit_transform_train(
        self, reference: pd.DataFrame, target: pd.Series
    ) -> pd.DataFrame:
        """Build training features with same-month targets excluded."""
        base = reference.reset_index(drop=True)
        target_values = pd.Series(target).reset_index(drop=True).astype(float)
        work = base[["MONTH_INDEX"]].copy()
        work["__target"] = target_values

        months = np.arange(
            int(base["MONTH_INDEX"].min()), int(base["MONTH_INDEX"].max()) + 1
        )
        global_monthly = (
            work.groupby("MONTH_INDEX")["__target"]
            .agg(["sum", "count"])
            .reindex(months, fill_value=0)
        )
        global_monthly["prior_sum"] = global_monthly["sum"].cumsum() - global_monthly["sum"]
        global_monthly["prior_count"] = (
            global_monthly["count"].cumsum() - global_monthly["count"]
        )
        global_monthly["prior_mean"] = (
            global_monthly["prior_sum"] / global_monthly["prior_count"].replace(0, np.nan)
        )
        row_global_prior = base["MONTH_INDEX"].map(global_monthly["prior_mean"])
        exact_monthly_mean = work.groupby("MONTH_INDEX")["__target"].mean()

        output = pd.DataFrame(index=base.index)
        output["monthly_target_lag"] = (
            base["MONTH_INDEX"] - self.lag_months
        ).map(exact_monthly_mean)
        output["monthly_target_lag"] = output["monthly_target_lag"].fillna(
            row_global_prior
        )

        global_recent_by_window: dict[int, pd.Series] = {}
        for window in HISTORY_WINDOWS:
            recent_sum = global_monthly["sum"].shift(1).rolling(
                window, min_periods=1
            ).sum()
            recent_count = global_monthly["count"].shift(1).rolling(
                window, min_periods=1
            ).sum()
            recent_mean = recent_sum / recent_count.replace(0, np.nan)
            global_recent_by_window[window] = recent_mean
            output[f"global_recent_{window}_mean"] = base["MONTH_INDEX"].map(
                recent_mean
            )
        output["global_recent_6_24_change"] = (
            output["global_recent_6_mean"] - output["global_recent_24_mean"]
        )

        for name, (columns, smoothing) in GROUP_SPECS.items():
            key_column = f"__key_{name}"
            group_work = base[["MONTH_INDEX"]].copy()
            group_work[key_column] = _group_key(base, columns)
            group_work["__target"] = target_values
            monthly = (
                group_work.groupby([key_column, "MONTH_INDEX"], as_index=False)[
                    "__target"
                ]
                .agg(["sum", "count"])
                .reset_index()
                .sort_values([key_column, "MONTH_INDEX"])
            )
            groups = group_work[key_column].drop_duplicates().to_numpy()
            grid = pd.MultiIndex.from_product(
                [groups, months], names=[key_column, "MONTH_INDEX"]
            ).to_frame(index=False)
            grid = grid.merge(monthly, on=[key_column, "MONTH_INDEX"], how="left")
            grid[["sum", "count"]] = grid[["sum", "count"]].fillna(0)
            grouped = grid.groupby(key_column, sort=False, observed=True)
            grid["prior_sum"] = grouped["sum"].cumsum() - grid["sum"]
            grid["prior_count"] = grouped["count"].cumsum() - grid["count"]
            row_keys = pd.MultiIndex.from_arrays(
                [_group_key(base, columns), base["MONTH_INDEX"]],
                names=[key_column, "MONTH_INDEX"],
            )
            lookup = grid.set_index([key_column, "MONTH_INDEX"])
            counts = lookup["prior_count"].reindex(row_keys).fillna(0).astype(float)
            sums = lookup["prior_sum"].reindex(row_keys).fillna(0).astype(float)
            output[f"{name}_history_mean"] = (
                sums.to_numpy() + smoothing * row_global_prior.to_numpy(dtype=float)
            ) / (counts.to_numpy() + smoothing)
            output[f"{name}_history_count"] = counts.to_numpy(dtype=float)

            for window in HISTORY_WINDOWS:
                grid[f"recent_sum_{window}"] = grouped["sum"].transform(
                    lambda values: values.shift(1).rolling(
                        window, min_periods=1
                    ).sum()
                )
                grid[f"recent_count_{window}"] = grouped["count"].transform(
                    lambda values: values.shift(1).rolling(
                        window, min_periods=1
                    ).sum()
                )
                lookup = grid.set_index([key_column, "MONTH_INDEX"])
                recent_sum = lookup[f"recent_sum_{window}"].reindex(row_keys).fillna(0)
                recent_count = (
                    lookup[f"recent_count_{window}"].reindex(row_keys).fillna(0)
                )
                global_recent = base["MONTH_INDEX"].map(
                    global_recent_by_window[window]
                ).fillna(row_global_prior)
                output[f"{name}_recent_{window}_mean"] = (
                    recent_sum.to_numpy(dtype=float)
                    + smoothing * global_recent.to_numpy(dtype=float)
                ) / (recent_count.to_numpy(dtype=float) + smoothing)
                output[f"{name}_recent_{window}_count"] = recent_count.to_numpy(
                    dtype=float
                )
            output[f"{name}_recent_6_24_change"] = (
                output[f"{name}_recent_6_mean"]
                - output[f"{name}_recent_24_mean"]
            )

        self.fit(base, target_values)
        return output[HISTORY_FEATURES]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Map reference history to rows, excluding their month and all later labels."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() or fit_transform_train() before transform().")
        target_frame = frame.reset_index(drop=True)
        target_keys = {
            name: _group_key(target_frame, columns)
            for name, (columns, _) in GROUP_SPECS.items()
        }
        output = pd.DataFrame(index=target_frame.index, columns=HISTORY_FEATURES, dtype=float)

        for month in sorted(target_frame["MONTH_INDEX"].unique()):
            row_mask = target_frame["MONTH_INDEX"].eq(month)
            row_indices = target_frame.index[row_mask]
            eligible = self.history_[self.history_["MONTH_INDEX"] < month]
            if eligible.empty:
                output.loc[row_indices, "monthly_target_lag"] = np.nan
                for window in HISTORY_WINDOWS:
                    output.loc[row_indices, f"global_recent_{window}_mean"] = np.nan
                output.loc[row_indices, "global_recent_6_24_change"] = np.nan
                for name in GROUP_SPECS:
                    output.loc[row_indices, f"{name}_history_mean"] = np.nan
                    output.loc[row_indices, f"{name}_history_count"] = 0.0
                    for window in HISTORY_WINDOWS:
                        output.loc[
                            row_indices, f"{name}_recent_{window}_mean"
                        ] = np.nan
                        output.loc[
                            row_indices, f"{name}_recent_{window}_count"
                        ] = 0.0
                    output.loc[
                        row_indices, f"{name}_recent_6_24_change"
                    ] = np.nan
                continue

            global_mean = float(eligible["__target"].mean())
            lag_value = self.monthly_mean_.get(month - self.lag_months, global_mean)
            output.loc[row_indices, "monthly_target_lag"] = float(lag_value)

            for window in HISTORY_WINDOWS:
                recent_global = eligible[
                    eligible["MONTH_INDEX"] >= month - window
                ]["__target"]
                value = float(recent_global.mean()) if not recent_global.empty else global_mean
                output.loc[row_indices, f"global_recent_{window}_mean"] = value
            output.loc[row_indices, "global_recent_6_24_change"] = (
                output.loc[row_indices, "global_recent_6_mean"]
                - output.loc[row_indices, "global_recent_24_mean"]
            )

            for name, (_, smoothing) in GROUP_SPECS.items():
                key_column = f"__key_{name}"
                stats = eligible.groupby(key_column)["__target"].agg(["sum", "count"])
                keys = target_keys[name].loc[row_indices]
                sums = keys.map(stats["sum"]).fillna(0).astype(float)
                counts = keys.map(stats["count"]).fillna(0).astype(float)
                means = (sums + smoothing * global_mean) / (counts + smoothing)
                output.loc[row_indices, f"{name}_history_mean"] = means.to_numpy()
                output.loc[row_indices, f"{name}_history_count"] = counts.to_numpy()

                for window in HISTORY_WINDOWS:
                    recent_eligible = eligible[
                        eligible["MONTH_INDEX"] >= month - window
                    ]
                    recent_stats = recent_eligible.groupby(key_column)["__target"].agg(
                        ["sum", "count"]
                    )
                    recent_global = (
                        float(recent_eligible["__target"].mean())
                        if not recent_eligible.empty
                        else global_mean
                    )
                    recent_sums = keys.map(recent_stats["sum"]).fillna(0).astype(float)
                    recent_counts = keys.map(recent_stats["count"]).fillna(0).astype(float)
                    recent_means = (
                        recent_sums + smoothing * recent_global
                    ) / (recent_counts + smoothing)
                    output.loc[
                        row_indices, f"{name}_recent_{window}_mean"
                    ] = recent_means.to_numpy()
                    output.loc[
                        row_indices, f"{name}_recent_{window}_count"
                    ] = recent_counts.to_numpy()
                output.loc[row_indices, f"{name}_recent_6_24_change"] = (
                    output.loc[row_indices, f"{name}_recent_6_mean"]
                    - output.loc[row_indices, f"{name}_recent_24_mean"]
                )

        return output[HISTORY_FEATURES]


def assemble_model_features(
    static_features: pd.DataFrame,
    history_features: pd.DataFrame,
    *,
    feature_profile: str = "core",
) -> pd.DataFrame:
    """Return the ordered feature matrix consumed by both model families."""
    if feature_profile not in MODEL_FEATURE_PROFILES:
        raise ValueError(
            f"Unknown feature profile {feature_profile!r}; "
            f"choose from {sorted(MODEL_FEATURE_PROFILES)}"
        )
    static = static_features.reset_index(drop=True)
    history = history_features.reset_index(drop=True)
    missing = set(BASE_FEATURES) - set(static)
    if missing:
        raise ValueError(f"Missing static features: {sorted(missing)}")
    if len(static) != len(history):
        raise ValueError("static and history feature lengths differ")
    base_columns = list(BASE_FEATURES)
    if static["median_rent_benchmark"].isna().all():
        base_columns.remove("median_rent_benchmark")
    history_columns = MODEL_FEATURE_PROFILES[feature_profile]
    expected_columns = base_columns + history_columns
    result = pd.concat([static[base_columns], history[history_columns]], axis=1)
    if result.columns.tolist() != expected_columns:
        raise RuntimeError("Unexpected model feature order")
    return result
