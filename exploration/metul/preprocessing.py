"""Data loading and feature engineering for the KDDM rental model."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "auxiliary/sg-coe-prices.csv",
    "auxiliary/sg-hdb-block.csv",
    "auxiliary/sg-mrt-stations.csv",
    "auxiliary/sg-schools.csv",
    "auxiliary/sg-shopping-malls.csv",
    "auxiliary/sg-stock-prices.csv",
]

HDB_VALUES = ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"]
LAT_SCALE = 111.0
LON_SCALE = 110.96
RENTAL_CATEGORICAL_COLUMNS = [
    "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FURNISHED",
]
HDB_CATEGORICAL_COLUMNS = [
    "TOWN", "BLOCK", "STREET", "POSTAL_CODE", "SUBZONE", "PLANNING_AREA", "REGION",
]
CATEGORICAL_CLEANING_DESCRIPTION = (
    "lowercase, trim, collapse whitespace, and replace hyphen/underscore/slash separators with spaces"
)


def _complete_data_dir(path):
    path = Path(path).expanduser()
    return path if all((path / file).exists() for file in REQUIRED_FILES) else None


def find_data_dir(data_dir=None):
    """Find a complete data directory on local or Colab filesystems."""
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    if os.environ.get("KDDM_DATA_DIR"):
        candidates.append(Path(os.environ["KDDM_DATA_DIR"]))

    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidates.append(parent / "data")

    for root in [Path("/content"), Path("/content/drive/MyDrive")]:
        candidates.extend([root / "data", root / "KDDM-project" / "data"])
        if root.exists():
            try:
                candidates.extend(path.parent for path in root.rglob("train.csv"))
            except (OSError, PermissionError):
                pass

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate).expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        complete = _complete_data_dir(candidate)
        if complete is not None:
            return complete

    raise FileNotFoundError(
        "Could not locate the complete data directory. Set KDDM_DATA_DIR "
        "or place the data folder on the active runtime."
    )


def load_datasets(data_dir=None):
    """Load train, test, and all auxiliary CSV files."""
    data_dir = find_data_dir(data_dir)
    aux_dir = data_dir / "auxiliary"
    datasets = {
        "train": pd.read_csv(data_dir / "train.csv"),
        "test": pd.read_csv(data_dir / "test.csv"),
        "coe": pd.read_csv(aux_dir / "sg-coe-prices.csv"),
        "hdb": pd.read_csv(aux_dir / "sg-hdb-block.csv"),
        "mrt": pd.read_csv(aux_dir / "sg-mrt-stations.csv"),
        "schools": pd.read_csv(aux_dir / "sg-schools.csv"),
        "malls": pd.read_csv(aux_dir / "sg-shopping-malls.csv"),
        "stock": pd.read_csv(aux_dir / "sg-stock-prices.csv"),
    }
    return data_dir, datasets


def categorical_clean(data, columns=None, *, context="categorical features", log=True):
    """Standardise categorical strings without fitting or restricting a category vocabulary.

    Values are lowercased, trimmed, whitespace-normalised, and have common
    separators folded to a space. Missing values remain missing, and unseen
    test categories remain present as their cleaned string values.
    """
    if isinstance(data, pd.Series):
        name = data.name if data.name is not None else "value"
        cleaned = categorical_clean(
            data.to_frame(name=name), columns=[name], context=context, log=log
        )
        return cleaned[name]

    result = data.copy()
    selected_columns = columns if columns is not None else result.columns
    for column in selected_columns:
        if column not in result.columns:
            continue

        values = result[column]
        before_unique = values.nunique(dropna=False)
        missing_count = values.isna().sum()
        cleaned = values.astype("string")
        cleaned = cleaned.str.lower().str.strip()
        cleaned = cleaned.str.replace(r"[-_/]+", " ", regex=True)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
        result[column] = cleaned
        after_unique = result[column].nunique(dropna=False)

        if len(result[column]) != len(values) or result[column].isna().sum() != missing_count:
            raise RuntimeError(f"Categorical cleaning changed row or missing-value count for {column}.")
        if log:
            print(
                f"[categorical_clean] {context} | {column}: "
                f"{before_unique} -> {after_unique} unique values | "
                f"{CATEGORICAL_CLEANING_DESCRIPTION}"
            )
    return result


def normalize_key(series):
    """Return the shared categorical-cleaned representation for lookup keys."""
    return categorical_clean(series, context="lookup key", log=False)


def _add_base_features(frame):
    result = categorical_clean(
        frame, columns=RENTAL_CATEGORICAL_COLUMNS, context="rental input"
    )
    approval_parts = result["RENT_APPROVAL_DATE"].str.split("-", expand=True)
    result["approval_year"] = approval_parts[0].astype(int)
    result["approval_month"] = approval_parts[1].astype(int)
    result["approval_date_num"] = result["approval_year"] * 12 + result["approval_month"]
    result["lease_age"] = result["approval_year"] - result["LEASE_COMMENCE_DATE"].astype(int)

    flat_type_order = {
        "1-room": 1, "1 room": 1, "2-room": 2, "2 room": 2,
        "3-room": 3, "3 room": 3, "4-room": 4, "4 room": 4,
        "5-room": 5, "5 room": 5, "executive": 6, "multi-generation": 7,
        "multi generation": 7,
    }
    result["flat_type_num"] = result["FLAT_TYPE"].str.lower().map(flat_type_order)

    # Verified constant in the supplied training data: FURNISHED is always
    # "yes" and FEE is always 0. AutoGluon consequently drops both columns as
    # useless; remove them here so every downstream model sees the same schema.
    return result.drop(columns=["FURNISHED", "FEE"], errors="ignore")


class RentalPreprocessor:
    """Feature preprocessor whose learned fallbacks are fit on training rows only."""

    def __init__(self, datasets):
        self.datasets = datasets
        self._is_fitted = False

        hdb = categorical_clean(
            datasets["hdb"], columns=HDB_CATEGORICAL_COLUMNS, context="HDB lookup"
        )
        hdb["BLOCK_KEY"] = normalize_key(hdb["BLOCK"])
        hdb["STREET_KEY"] = normalize_key(hdb["STREET"])
        hdb["TOWN_KEY"] = normalize_key(hdb["TOWN"])
        self.hdb_lookup_ = hdb.drop_duplicates(["BLOCK_KEY", "STREET_KEY"])[
            ["BLOCK_KEY", "STREET_KEY", *HDB_VALUES, "REGION"]
        ]

    @staticmethod
    def _mode_or_nan(values):
        modes = values.dropna().mode()
        return modes.iloc[0] if not modes.empty else np.nan

    def _merge_hdb_lookup(self, frame):
        result = frame.copy()
        result["BLOCK_KEY"] = normalize_key(result["BLOCK"])
        result["STREET_KEY"] = normalize_key(result["STREET"])
        result["TOWN_KEY"] = normalize_key(result["TOWN"])
        return result.merge(
            self.hdb_lookup_, on=["BLOCK_KEY", "STREET_KEY"], how="left"
        )

    def fit(self, train_frame):
        """Learn HDB fallback statistics from the training partition only."""
        train = self._merge_hdb_lookup(_add_base_features(train_frame))
        # Freeze exact block/street attributes to locations encountered during
        # fitting. Transform never consults validation or test rows to create a
        # new lookup entry; unseen locations fall through to the train-derived
        # street, town, and global fallback statistics below.
        self.hdb_lookup_ = train[
            ["BLOCK_KEY", "STREET_KEY", *HDB_VALUES, "REGION"]
        ].drop_duplicates(["BLOCK_KEY", "STREET_KEY"])
        self.street_medians_ = train.groupby(
            ["TOWN_KEY", "STREET_KEY"]
        )[HDB_VALUES].median()
        self.town_medians_ = train.groupby("TOWN_KEY")[HDB_VALUES].median()
        self.global_medians_ = train[HDB_VALUES].median()
        self.town_regions_ = train.groupby("TOWN_KEY")["REGION"].agg(
            self._mode_or_nan
        )
        self.global_region_ = self._mode_or_nan(train["REGION"])
        self._is_fitted = True
        return self

    def _add_hdb_features(self, frame):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        result = self._merge_hdb_lookup(_add_base_features(frame))
        street_index = pd.MultiIndex.from_arrays(
            [result["TOWN_KEY"], result["STREET_KEY"]]
        )

        for column in HDB_VALUES:
            street_fill = pd.Series(
                street_index.map(self.street_medians_[column]), index=result.index
            )
            result[column] = result[column].fillna(street_fill)
            result[column] = result[column].fillna(
                result["TOWN_KEY"].map(self.town_medians_[column])
            )
            result[column] = result[column].fillna(self.global_medians_[column])

        result["REGION"] = result["REGION"].fillna(
            result["TOWN_KEY"].map(self.town_regions_)
        )
        result["REGION"] = result["REGION"].fillna(self.global_region_)
        result = categorical_clean(
            result, columns=["REGION"], context="derived HDB feature"
        )
        return result.drop(columns=["BLOCK_KEY", "STREET_KEY", "TOWN_KEY"])

    def transform(self, frame):
        """Transform a validation or test partition without learning from it."""
        result = self._add_hdb_features(frame)
        result = _add_spatial_features(
            result, self.datasets["mrt"], self.datasets["malls"], self.datasets["schools"]
        )
        return _add_temporal_features(result, self.datasets["coe"], self.datasets["stock"])

    def fit_transform(self, train_frame):
        return self.fit(train_frame).transform(train_frame)


def _proximity_features(facility_df, location_coords_km, prefix):
    facility_coords = facility_df[["LATITUDE", "LONGITUDE"]].dropna().to_numpy()
    tree = cKDTree(facility_coords * [LAT_SCALE, LON_SCALE])
    nearest_distance, _ = tree.query(location_coords_km, k=1)
    return pd.DataFrame({
        f"{prefix}_dist_nearest_km": nearest_distance,
        f"{prefix}_within_1km": [len(i) for i in tree.query_ball_point(location_coords_km, r=1.0)],
        f"{prefix}_within_2km": [len(i) for i in tree.query_ball_point(location_coords_km, r=2.0)],
    })


def _add_spatial_features(frame, mrt_df, malls_df, schools_df):
    result = frame.copy()
    locations = result[["BLOCK", "STREET", "LATITUDE", "LONGITUDE"]].drop_duplicates().reset_index(drop=True)
    location_coords_km = locations[["LATITUDE", "LONGITUDE"]].to_numpy() * [LAT_SCALE, LON_SCALE]
    spatial = pd.concat([
        locations[["BLOCK", "STREET"]],
        _proximity_features(mrt_df, location_coords_km, "mrt"),
        _proximity_features(malls_df, location_coords_km, "mall"),
        _proximity_features(schools_df, location_coords_km, "school"),
    ], axis=1)
    return result.merge(spatial, on=["BLOCK", "STREET"], how="left")


def _add_temporal_features(frame, coe_df, stock_df):
    result = frame.copy()
    month_map = {month: number for number, month in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )}

    coe = coe_df.copy()
    coe["month_num"] = coe["MONTH"].str.lower().map(month_map)
    for column in ["QUOTA_PREMIUM", "BIDS_RECEIVED"]:
        coe[column] = coe[column].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    coe_monthly = coe.groupby(["YEAR", "month_num"], as_index=False).agg(
        coe_quota_premium_mean=("QUOTA_PREMIUM", "mean"),
        coe_bids_mean=("BIDS_RECEIVED", "mean"),
    )
    result = result.merge(
        coe_monthly,
        left_on=["approval_year", "approval_month"],
        right_on=["YEAR", "month_num"],
        how="left",
    ).drop(columns=["YEAR", "month_num"])

    stock = stock_df.copy()
    stock["DATE"] = pd.to_datetime(stock["DATE"])
    stock["YEAR"] = stock["DATE"].dt.year
    stock["month_num"] = stock["DATE"].dt.month
    stock_monthly = stock.groupby(["YEAR", "month_num"], as_index=False).agg(
        stock_close_mean=("CLOSE", "mean"),
        stock_volume_mean=("VOLUME", "mean"),
    )
    return result.merge(
        stock_monthly,
        left_on=["approval_year", "approval_month"],
        right_on=["YEAR", "month_num"],
        how="left",
    ).drop(columns=["YEAR", "month_num"])


def preprocess_frame(frame, datasets, preprocessor=None, fit=False):
    """Transform one frame with a fitted preprocessor.

    Set ``fit=True`` only for the training partition. Validation and test frames
    must be passed with the same fitted ``preprocessor`` and ``fit=False``.
    """
    if preprocessor is None:
        preprocessor = RentalPreprocessor(datasets)
        if not fit:
            raise ValueError(
                "A fitted preprocessor is required for transform-only calls. "
                "Use preprocessor.fit_transform(train) first."
            )
    return preprocessor.fit_transform(frame) if fit else preprocessor.transform(frame)


def preprocess_train_test(data_dir=None):
    """Load all CSVs and return data path, datasets, joined train, and joined test."""
    data_dir, datasets = load_datasets(data_dir)
    preprocessor = RentalPreprocessor(datasets)
    train = preprocessor.fit_transform(datasets["train"])
    test = preprocessor.transform(datasets["test"])
    return data_dir, datasets, train, test
