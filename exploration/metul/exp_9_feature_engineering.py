"""Leakage-safe feature preparation shared by Experiment 9 model notebooks."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


TARGET = "MONTHLY_RENT"
GROUP_COLUMNS = ["TOWN", "FLAT_TYPE"]
COMPACT_FEATURES = [
    "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL",
    "FLOOR_AREA_SQM", "approval_date_num", "approval_month",
    "coe_quota_premium_mean", "YEAR_COMPLETED", "MAX_FLOOR",
    "mrt_dist_nearest_km", "mall_dist_nearest_km",
]
NEW_FEATURES = [
    "monthly_target_lag", "town_flat_type_target_mean", "nearest_mrt_name",
]
ALL_FEATURES = COMPACT_FEATURES + NEW_FEATURES


class TargetFeatureTransformer:
    """Build monthly-lag and town/flat-type target features from labelled history."""

    def __init__(self, lag_months, group_columns=GROUP_COLUMNS, target=TARGET):
        self.lag_months = lag_months
        self.group_columns = group_columns
        self.target = target

    def _keys(self, frame):
        return frame[self.group_columns + ["period"]].copy()

    def fit(self, labelled_frame):
        history = labelled_frame[self.group_columns + ["period", self.target]].copy()
        self.global_mean_ = history[self.target].mean()
        self.group_mean_ = history.groupby(self.group_columns)[self.target].mean()
        self.monthly_target_mean_ = history.groupby("period")[self.target].mean()
        return self

    def transform(self, frame):
        """Transform validation/test rows using the labelled history supplied to fit()."""
        result = self._keys(frame)
        result["monthly_target_lag"] = (
            result["period"] - self.lag_months
        ).map(self.monthly_target_mean_).fillna(self.global_mean_)
        group_mean = pd.Series(
            result.set_index(self.group_columns).index.map(self.group_mean_), index=result.index
        )
        result["town_flat_type_target_mean"] = group_mean.fillna(self.global_mean_)
        return result[["monthly_target_lag", "town_flat_type_target_mean"]]

    def fit_transform_train(self, labelled_frame):
        """Transform labelled training rows with strictly earlier-month target means."""
        self.fit(labelled_frame)
        history = labelled_frame[self.group_columns + ["period", self.target]].copy()
        monthly = (
            history.groupby(self.group_columns + ["period"], as_index=False)[self.target]
            .agg(["sum", "count"]).reset_index()
            .sort_values(self.group_columns + ["period"])
        )
        monthly["prior_sum"] = monthly.groupby(self.group_columns)["sum"].cumsum() - monthly["sum"]
        monthly["prior_count"] = monthly.groupby(self.group_columns)["count"].cumsum() - monthly["count"]
        monthly["town_flat_type_target_mean"] = (
            monthly["prior_sum"] / monthly["prior_count"].replace(0, np.nan)
        )

        global_monthly = history.groupby("period")[self.target].agg(["sum", "count"]).sort_index()
        global_monthly["prior_sum"] = global_monthly["sum"].cumsum() - global_monthly["sum"]
        global_monthly["prior_count"] = global_monthly["count"].cumsum() - global_monthly["count"]
        global_prior_mean = global_monthly["prior_sum"] / global_monthly["prior_count"].replace(0, np.nan)

        result = self._keys(labelled_frame).merge(
            monthly[self.group_columns + ["period", "town_flat_type_target_mean"]],
            on=self.group_columns + ["period"], how="left",
        )
        result["town_flat_type_target_mean"] = result["town_flat_type_target_mean"].fillna(
            result["period"].map(global_prior_mean)
        )
        result["monthly_target_lag"] = (result["period"] - self.lag_months).map(
            self.monthly_target_mean_
        ).fillna(result["period"].map(global_prior_mean))
        return result[["monthly_target_lag", "town_flat_type_target_mean"]]


def preprocess_partition(preprocessor, raw_frame, fit=False):
    features = raw_frame.drop(columns=["period", TARGET], errors="ignore")
    return preprocessor.fit_transform(features) if fit else preprocessor.transform(features)


def nearest_mrt_name(base_frame, mrt_frame):
    """Return the supplied MRT station name nearest to each preprocessed row."""
    stations = mrt_frame[["NAME", "LATITUDE", "LONGITUDE"]].dropna().copy()
    station_coords_km = stations[["LATITUDE", "LONGITUDE"]].to_numpy() * [111.0, 110.96]
    row_coords_km = base_frame[["LATITUDE", "LONGITUDE"]].to_numpy() * [111.0, 110.96]
    _, nearest_indices = cKDTree(station_coords_km).query(row_coords_km, k=1)
    return pd.Series(
        stations["NAME"].astype(object).to_numpy()[nearest_indices],
        index=base_frame.index, name="nearest_mrt_name", dtype=object,
    )


def labelled_history(raw_frame, base_frame):
    result = base_frame[GROUP_COLUMNS].copy().reset_index(drop=True)
    result["period"] = raw_frame["period"].to_numpy()
    result[TARGET] = raw_frame[TARGET].to_numpy()
    return result


def unlabelled_history(raw_frame, base_frame):
    result = base_frame[GROUP_COLUMNS].copy().reset_index(drop=True)
    result["period"] = raw_frame["period"].to_numpy()
    return result


def make_model_frame(base_frame, target_features, mrt_frame):
    return pd.concat(
        [
            base_frame[COMPACT_FEATURES].reset_index(drop=True),
            target_features.reset_index(drop=True),
            nearest_mrt_name(base_frame, mrt_frame).reset_index(drop=True),
        ],
        axis=1,
    )
