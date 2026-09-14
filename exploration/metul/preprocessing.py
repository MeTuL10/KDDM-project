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


def normalize_key(series):
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def _add_base_features(frame):
    result = frame.copy()
    approval_parts = result["RENT_APPROVAL_DATE"].str.split("-", expand=True)
    result["approval_year"] = approval_parts[0].astype(int)
    result["approval_month"] = approval_parts[1].astype(int)
    result["approval_date_num"] = result["approval_year"] * 12 + result["approval_month"]
    result["lease_age"] = result["approval_year"] - result["LEASE_COMMENCE_DATE"].astype(int)

    flat_type_order = {
        "1-room": 1, "1 room": 1, "2-room": 2, "2 room": 2,
        "3-room": 3, "3 room": 3, "4-room": 4, "4 room": 4,
        "5-room": 5, "5 room": 5, "executive": 6, "multi-generation": 7,
    }
    result["flat_type_num"] = result["FLAT_TYPE"].str.lower().map(flat_type_order)
    return result


def _add_hdb_features(frame, hdb_df):
    result = frame.copy()
    result["BLOCK_KEY"] = normalize_key(result["BLOCK"])
    result["STREET_KEY"] = normalize_key(result["STREET"])
    result["TOWN_KEY"] = normalize_key(result["TOWN"])

    hdb = hdb_df.copy()
    hdb["BLOCK_KEY"] = normalize_key(hdb["BLOCK"])
    hdb["STREET_KEY"] = normalize_key(hdb["STREET"])
    hdb["TOWN_KEY"] = normalize_key(hdb["TOWN"])

    lookup = hdb.drop_duplicates(["BLOCK_KEY", "STREET_KEY"])
    result = result.merge(
        lookup[["BLOCK_KEY", "STREET_KEY", *HDB_VALUES]],
        on=["BLOCK_KEY", "STREET_KEY"],
        how="left",
    )

    street_medians = hdb.groupby(["TOWN_KEY", "STREET_KEY"])[HDB_VALUES].median()
    town_medians = hdb.groupby("TOWN_KEY")[HDB_VALUES].median()
    street_index = pd.MultiIndex.from_arrays([result["TOWN_KEY"], result["STREET_KEY"]])

    for column in HDB_VALUES:
        street_fill = pd.Series(street_index.map(street_medians[column]), index=result.index)
        result[column] = result[column].fillna(street_fill)
        result[column] = result[column].fillna(result["TOWN_KEY"].map(town_medians[column]))

    return result.drop(columns=["BLOCK_KEY", "STREET_KEY", "TOWN_KEY"])


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


def preprocess_frame(frame, datasets):
    """Apply the same feature engineering to a train or test frame."""
    result = _add_base_features(frame)
    result = _add_hdb_features(result, datasets["hdb"])
    result = _add_spatial_features(result, datasets["mrt"], datasets["malls"], datasets["schools"])
    return _add_temporal_features(result, datasets["coe"], datasets["stock"])


def preprocess_train_test(data_dir=None):
    """Load all CSVs and return data path, datasets, joined train, and joined test."""
    data_dir, datasets = load_datasets(data_dir)
    train = preprocess_frame(datasets["train"], datasets)
    test = preprocess_frame(datasets["test"], datasets)
    return data_dir, datasets, train, test

