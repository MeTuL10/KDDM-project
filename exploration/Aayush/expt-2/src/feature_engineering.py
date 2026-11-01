"""
Feature Engineering Pipeline (Phases 3 & 4).
Generates:
1. Phase 3: Spatial features via BallTree / Haversine (CBD, MRT, Shopping Malls, Schools)
2. Phase 4: Macroeconomic & Extrapolation features (HDB Median Rent, COE, Stock Indices)
3. Domain interactions & non-linear proximity scores
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from typing import Tuple, Dict, Any

try:
    from . import config
except ImportError:
    import config

def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Compute spherical haversine distance in kilometers between array of coords and a fixed point."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return config.EARTH_RADIUS_KM * c

def build_spatial_features(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute geospatial proximity metrics on unique block coordinates.
    Evaluated on unique blocks for optimal computational efficiency.
    """
    print("   [Spatial FE] Calculating spatial proximity metrics on unique blocks...")
    spatial = blocks_df[["town_clean", "block_clean", "street_clean", "LATITUDE", "LONGITUDE"]].drop_duplicates().copy()
    block_coords_rad = np.radians(spatial[["LATITUDE", "LONGITUDE"]].values)

    # 1. Distance to Downtown Core / CBD (Raffles Place)
    spatial["dist_to_cbd"] = haversine_distance(
        spatial["LATITUDE"].values, spatial["LONGITUDE"].values,
        config.CBD_LAT, config.CBD_LON
    )

    # 2. MRT Station Features (sg-mrt-stations.csv)
    mrt = pd.read_csv(config.MRT_STATIONS_PATH)
    open_mrt = mrt[mrt["STATUS"].str.strip().str.lower() == "open"]
    planned_mrt = mrt[mrt["STATUS"].str.strip().str.lower() == "planned"]

    open_mrt_rad = np.radians(open_mrt[["LATITUDE", "LONGITUDE"]].values)
    open_mrt_tree = BallTree(open_mrt_rad, metric="haversine")

    dist_mrt, _ = open_mrt_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_mrt"] = dist_mrt[:, 0] * config.EARTH_RADIUS_KM
    spatial["mrt_within_500m"] = open_mrt_tree.query_radius(
        block_coords_rad, r=0.5 / config.EARTH_RADIUS_KM, count_only=True
    )
    spatial["mrt_within_1km"] = open_mrt_tree.query_radius(
        block_coords_rad, r=1.0 / config.EARTH_RADIUS_KM, count_only=True
    )

    # Planned MRT proximity
    planned_mrt_rad = np.radians(planned_mrt[["LATITUDE", "LONGITUDE"]].values)
    planned_tree = BallTree(planned_mrt_rad, metric="haversine")
    dist_planned, _ = planned_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_planned_mrt"] = dist_planned[:, 0] * config.EARTH_RADIUS_KM

    # 3. Shopping Malls (sg-shopping-malls.csv)
    malls = pd.read_csv(config.SHOPPING_MALLS_PATH)
    malls_rad = np.radians(malls[["LATITUDE", "LONGITUDE"]].values)
    malls_tree = BallTree(malls_rad, metric="haversine")

    dist_mall, _ = malls_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_mall"] = dist_mall[:, 0] * config.EARTH_RADIUS_KM
    spatial["malls_within_1km"] = malls_tree.query_radius(
        block_coords_rad, r=1.0 / config.EARTH_RADIUS_KM, count_only=True
    )
    spatial["malls_within_2km"] = malls_tree.query_radius(
        block_coords_rad, r=2.0 / config.EARTH_RADIUS_KM, count_only=True
    )

    # 4. Schools (sg-schools.csv)
    schools = pd.read_csv(config.SCHOOLS_PATH)
    
    # Primary schools
    is_primary = schools["MAINLEVEL_CODE"].astype(str).str.lower().str.contains("primary")
    primary_schools = schools[is_primary]
    primary_rad = np.radians(primary_schools[["LATITUDE", "LONGITUDE"]].values)
    primary_tree = BallTree(primary_rad, metric="haversine")

    dist_pri, _ = primary_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_primary_school"] = dist_pri[:, 0] * config.EARTH_RADIUS_KM
    spatial["primary_schools_within_1km"] = primary_tree.query_radius(
        block_coords_rad, r=1.0 / config.EARTH_RADIUS_KM, count_only=True
    )

    # Secondary schools
    is_secondary = schools["MAINLEVEL_CODE"].astype(str).str.lower().str.contains("secondary")
    secondary_schools = schools[is_secondary]
    sec_rad = np.radians(secondary_schools[["LATITUDE", "LONGITUDE"]].values)
    sec_tree = BallTree(sec_rad, metric="haversine")

    dist_sec, _ = sec_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_secondary_school"] = dist_sec[:, 0] * config.EARTH_RADIUS_KM

    # Non-linear proximity scores (inverse distance with smoothing)
    spatial["mrt_proximity_score"] = 1.0 / (spatial["dist_nearest_mrt"] + 0.1)
    spatial["mall_proximity_score"] = 1.0 / (spatial["dist_nearest_mall"] + 0.1)
    spatial["school_proximity_score"] = 1.0 / (spatial["dist_nearest_primary_school"] + 0.1)

    print(f"   [Spatial FE] Computed {len(spatial.columns) - 5} spatial features for {len(spatial)} blocks.")
    return spatial.drop(columns=["LATITUDE", "LONGITUDE"])

def build_median_rent_benchmark() -> pd.DataFrame:
    """
    Extract and impute official quarterly HDB median rent benchmarks.
    Handles town-level nulls with national flat-type medians and forward-fills future test quarters.
    """
    print("   [Macro FE] Building quarterly HDB median rent benchmarks...")
    mr = pd.read_csv(config.HDB_MEDIAN_RENT_PATH)
    mr["town_clean"] = mr["town"].astype(str).str.strip().str.lower()
    mr["flat_type_clean"] = mr["flat_type"].map({
        "1-RM": "1-room", "2-RM": "2-room", "3-RM": "3-room",
        "4-RM": "4-room", "5-RM": "5-room", "EXEC": "executive"
    })
    mr["median_rent_num"] = pd.to_numeric(mr["median_rent"], errors="coerce")

    # Quarterly national median by flat type
    nat_mr = mr.groupby(["quarter", "flat_type_clean"])["median_rent_num"].median().reset_index()
    nat_mr.rename(columns={"median_rent_num": "nat_median_rent_benchmark"}, inplace=True)

    # Fill rare 1-room missing with ~65% of 2-room median rent
    nat_pivot = nat_mr.pivot(index="quarter", columns="flat_type_clean", values="nat_median_rent_benchmark")
    if "1-room" not in nat_pivot.columns or nat_pivot["1-room"].isnull().all():
        nat_pivot["1-room"] = nat_pivot["2-room"] * 0.65
    else:
        nat_pivot["1-room"] = nat_pivot["1-room"].fillna(nat_pivot["2-room"] * 0.65)
    
    # Forward fill 2-room in any rare quarter if missing
    nat_pivot["2-room"] = nat_pivot["2-room"].ffill().bfill()
    nat_pivot["1-room"] = nat_pivot["1-room"].fillna(nat_pivot["2-room"] * 0.65)
    
    nat_mr_filled = nat_pivot.reset_index().melt(
        id_vars="quarter", value_name="nat_median_rent_benchmark"
    )

    # Merge national benchmark back to town level
    full_mr = mr.merge(nat_mr_filled, on=["quarter", "flat_type_clean"], how="right")
    full_mr["median_rent_benchmark"] = full_mr["median_rent_num"].combine_first(
        full_mr["nat_median_rent_benchmark"]
    )

    # Forward fill 2026-Q3 using 2026-Q2
    q2_2026 = full_mr[full_mr["quarter"] == "2026-Q2"].copy()
    q2_2026["quarter"] = "2026-Q3"
    full_mr = pd.concat([full_mr, q2_2026], ignore_index=True)

    keep_cols = ["quarter", "town_clean", "flat_type_clean", "median_rent_benchmark", "nat_median_rent_benchmark"]
    clean_mr = full_mr[keep_cols].drop_duplicates(subset=["quarter", "town_clean", "flat_type_clean"])
    
    # Calculate estate premium ratio
    clean_mr["median_rent_ratio"] = clean_mr["median_rent_benchmark"] / clean_mr["nat_median_rent_benchmark"]
    return clean_mr

def build_coe_features() -> pd.DataFrame:
    """
    Extract monthly average Certificate of Entitlement (COE) quota premiums.
    Serves as an indicator of Singapore macroeconomic liquidity and price levels.
    """
    print("   [Macro FE] Building monthly COE price features...")
    coe = pd.read_csv(config.COE_PRICES_PATH)
    
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    coe["approval_month"] = coe["MONTH"].astype(str).str.strip().str.lower().map(month_map)
    coe["approval_year"] = coe["YEAR"].astype(int)
    coe["quota_num"] = coe["QUOTA_PREMIUM"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    
    # Monthly averages by category
    coe_pvt = coe.groupby(["approval_year", "approval_month", "CATEGORY"])["quota_num"].mean().unstack()
    coe_pvt = coe_pvt.rename(columns={
        "a": "coe_cat_a",
        "b": "coe_cat_b",
        "c": "coe_cat_c",
        "e": "coe_cat_e"
    }).reset_index()

    # Fill any sparse category with row mean
    coe_pvt["coe_overall_mean"] = coe_pvt[["coe_cat_a", "coe_cat_b", "coe_cat_e"]].mean(axis=1)
    coe_pvt["coe_cat_a"] = coe_pvt["coe_cat_a"].fillna(coe_pvt["coe_overall_mean"])
    coe_pvt["coe_cat_b"] = coe_pvt["coe_cat_b"].fillna(coe_pvt["coe_overall_mean"])
    coe_pvt["coe_premium_spread"] = coe_pvt["coe_cat_b"] - coe_pvt["coe_cat_a"]
    
    return coe_pvt[["approval_year", "approval_month", "coe_cat_a", "coe_cat_b", "coe_overall_mean", "coe_premium_spread"]]

def build_stock_features() -> pd.DataFrame:
    """
    Extract monthly Singapore banking and blue-chip stock index signals.
    """
    print("   [Macro FE] Building monthly stock liquidity and index features...")
    stocks = pd.read_csv(config.STOCK_PRICES_PATH)
    stocks["date"] = pd.to_datetime(stocks["DATE"])
    stocks["approval_year"] = stocks["date"].dt.year
    stocks["approval_month"] = stocks["date"].dt.month

    top_banks = ["DBS Group", "OCBC Bank", "UOB", "Singtel"]
    bank_df = stocks[stocks["NAME"].isin(top_banks)].groupby(
        ["approval_year", "approval_month", "NAME"]
    )["CLOSE"].mean().unstack().reset_index()

    bank_df = bank_df.rename(columns={
        "DBS Group": "stock_dbs_close",
        "OCBC Bank": "stock_ocbc_close",
        "UOB": "stock_uob_close",
        "Singtel": "stock_singtel_close"
    })

    # Weighted banking & financial liquidity index
    bank_df["sg_financial_index"] = (
        bank_df["stock_dbs_close"] +
        bank_df["stock_ocbc_close"] * 2.0 +
        bank_df["stock_uob_close"] +
        bank_df["stock_singtel_close"] * 5.0
    ) / 4.0

    return bank_df[["approval_year", "approval_month", "stock_dbs_close", "stock_ocbc_close", "stock_uob_close", "sg_financial_index"]]

def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full feature engineering pipeline on cleaned train and test datasets.
    """
    print("=" * 80)
    print("PHASES 3 & 4: EXECUTING FEATURE ENGINEERING PIPELINE")
    print("=" * 80)

    # 1. Spatial Feature Enrichment
    all_blocks = pd.concat([train_df, test_df], ignore_index=True)
    spatial_df = build_spatial_features(all_blocks)

    print("1. Merging spatial features with Train and Test...")
    train_feat = train_df.merge(
        spatial_df, on=["town_clean", "block_clean", "street_clean"], how="left"
    )
    test_feat = test_df.merge(
        spatial_df, on=["town_clean", "block_clean", "street_clean"], how="left"
    )

    # 2. Macroeconomic Enrichment: Median Rent Benchmark
    median_rent_df = build_median_rent_benchmark()
    print("2. Merging quarterly median rent benchmarks...")
    train_feat = train_feat.merge(
        median_rent_df,
        left_on=["approval_quarter", "town_clean", "flat_type_clean"],
        right_on=["quarter", "town_clean", "flat_type_clean"],
        how="left"
    ).drop(columns=["quarter"])

    test_feat = test_feat.merge(
        median_rent_df,
        left_on=["approval_quarter", "town_clean", "flat_type_clean"],
        right_on=["quarter", "town_clean", "flat_type_clean"],
        how="left"
    ).drop(columns=["quarter"])

    # Fallback for any unmapped town: use national benchmark
    train_feat["median_rent_benchmark"] = train_feat["median_rent_benchmark"].combine_first(
        train_feat["nat_median_rent_benchmark"]
    )
    test_feat["median_rent_benchmark"] = test_feat["median_rent_benchmark"].combine_first(
        test_feat["nat_median_rent_benchmark"]
    )
    train_feat["median_rent_ratio"] = train_feat["median_rent_ratio"].fillna(1.0)
    test_feat["median_rent_ratio"] = test_feat["median_rent_ratio"].fillna(1.0)

    # 3. Macroeconomic Enrichment: COE Prices
    coe_df = build_coe_features()
    print("3. Merging monthly COE quota premiums...")
    train_feat = train_feat.merge(coe_df, on=["approval_year", "approval_month"], how="left")
    test_feat = test_feat.merge(coe_df, on=["approval_year", "approval_month"], how="left")

    # 4. Macroeconomic Enrichment: Stock Index
    stock_df = build_stock_features()
    print("4. Merging monthly stock indices...")
    train_feat = train_feat.merge(stock_df, on=["approval_year", "approval_month"], how="left")
    test_feat = test_feat.merge(stock_df, on=["approval_year", "approval_month"], how="left")

    # 5. Domain Interaction Features
    print("5. Computing domain interaction features...")
    for df in [train_feat, test_feat]:
        # Unit area per room
        df["area_per_room"] = df["FLOOR_AREA_SQM"] / df["room_count"]
        # Space-CBD premium interactions
        df["cbd_room_interaction"] = df["dist_to_cbd"] * df["room_count"]
        df["cbd_area_interaction"] = df["dist_to_cbd"] * df["FLOOR_AREA_SQM"]
        # Floor level & area interaction
        df["floor_area_interaction"] = df["MAX_FLOOR"] * df["FLOOR_AREA_SQM"]
        # Macro benchmark relative deviation
        df["area_to_benchmark_ratio"] = df["FLOOR_AREA_SQM"] / (df["median_rent_benchmark"] / 30.0)

    # 6. Verification & Quality Assertions
    print("6. Verifying feature matrix integrity...")
    assert len(test_feat) == 50000, f"Test count changed: {len(test_feat)} != 50000"
    
    # Assert zero nulls across engineered features
    train_nulls = train_feat.isnull().sum()
    test_nulls = test_feat.isnull().sum()
    assert train_nulls.sum() == 0, f"Found nulls in train features:\n{train_nulls[train_nulls > 0]}"
    assert test_nulls.sum() == 0, f"Found nulls in test features:\n{test_nulls[test_nulls > 0]}"

    # Save to disk
    os.makedirs(config.FEATURE_DATA_DIR, exist_ok=True)
    train_feat_path = os.path.join(config.FEATURE_DATA_DIR, "train_features.parquet")
    test_feat_path = os.path.join(config.FEATURE_DATA_DIR, "test_features.parquet")

    train_feat.to_parquet(train_feat_path, index=False)
    test_feat.to_parquet(test_feat_path, index=False)
    print(f"   Saved train features: {train_feat_path} ({train_feat.shape})")
    print(f"   Saved test features:  {test_feat_path} ({test_feat.shape})")

    # Save metadata
    meta = {
        "n_train_rows": len(train_feat),
        "n_test_rows": len(test_feat),
        "feature_count": train_feat.shape[1] - 1, # excluding target
        "features": [c for c in train_feat.columns if c != "MONTHLY_RENT"]
    }
    with open(os.path.join(config.FEATURE_DATA_DIR, "feature_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nFeature Engineering (Phases 3 & 4) completed successfully!")
    return train_feat, test_feat

if __name__ == "__main__":
    clean_train = pd.read_parquet(os.path.join(config.CLEANED_DATA_DIR, "train_cleaned.parquet"))
    clean_test = pd.read_parquet(os.path.join(config.CLEANED_DATA_DIR, "test_cleaned.parquet"))
    engineer_features(clean_train, clean_test)

