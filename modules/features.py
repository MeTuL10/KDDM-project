"""
Feature Engineering Module v2 (Sprint 1 & Sprint 2 Enhancements)
Path: modules/features.py (ported from expt-2/src/feature_engineering_v2.py)

Key Advancements:
1. Dynamic Macro Extrapolation: Extrapolates 2026-Q3 median rent benchmark using local quarterly growth.
2. High-Resolution Spatial Amenities:
   - MRT multi-line interchanges (distance and count within 1km).
   - Top-tier / independent schools (distance and count within 1km).
   - Structural ratios: floor_area_per_room, floor_area_per_max_floor, flat_age_squared.
3. Hierarchical Out-Of-Fold Bayesian Target Encoding:
   - Multi-level empirical Bayes shrinkage across (Town -> Block -> Block x Flat Type -> Street -> Subzone).
   - Zero leakage: Strictly 5-Fold Out-of-Fold (OOF) on train, global historical priors on test.
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold
import config

EARTH_RADIUS_KM = 6371.0

def build_spatial_features_v2(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes spatial proximity features including MRT interchanges and top-tier schools.
    """
    print("   [Spatial FE v2] Building BallTree spatial features for unique blocks...")
    spatial = block_df[["block_id", "LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"]].copy()
    block_coords = spatial[["LATITUDE", "LONGITUDE"]].values
    block_coords_rad = np.radians(block_coords)

    # 1. Distance to Singapore CBD (Downtown Core: 1.2838° N, 103.8591° E)
    cbd_lat, cbd_lon = 1.2838, 103.8591
    dlat = block_coords_rad[:, 0] - np.radians(cbd_lat)
    dlon = block_coords_rad[:, 1] - np.radians(cbd_lon)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(cbd_lat)) * np.cos(block_coords_rad[:, 0]) * np.sin(dlon / 2.0)**2
    spatial["dist_cbd"] = EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(a))

    # 2. MRT Stations
    mrt = pd.read_csv(config.MRT_STATIONS_PATH)
    mrt_coords_rad = np.radians(mrt[["LATITUDE", "LONGITUDE"]].values)
    mrt_tree = BallTree(mrt_coords_rad, metric="haversine")

    dist_mrt, _ = mrt_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_mrt"] = dist_mrt[:, 0] * EARTH_RADIUS_KM
    spatial["mrt_within_500m"] = mrt_tree.query_radius(block_coords_rad, r=0.5 / EARTH_RADIUS_KM, count_only=True)
    spatial["mrt_within_1km"] = mrt_tree.query_radius(block_coords_rad, r=1.0 / EARTH_RADIUS_KM, count_only=True)

    # Operating vs Planned MRT
    is_open = mrt["STATUS"].astype(str).str.lower() == "open"
    open_tree = BallTree(np.radians(mrt.loc[is_open, ["LATITUDE", "LONGITUDE"]].values), metric="haversine")
    dist_open_mrt, _ = open_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_operating_mrt"] = dist_open_mrt[:, 0] * EARTH_RADIUS_KM

    planned_tree = BallTree(np.radians(mrt.loc[~is_open, ["LATITUDE", "LONGITUDE"]].values), metric="haversine")
    dist_plan_mrt, _ = planned_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_planned_mrt"] = dist_plan_mrt[:, 0] * EARTH_RADIUS_KM

    # 3. MRT Multi-Line Interchanges (NEW)
    station_counts = mrt.groupby("NAME")["CODE"].nunique()
    interchange_names = set(station_counts[station_counts > 1].index)
    interchanges = mrt[mrt["NAME"].isin(interchange_names)].drop_duplicates(subset=["NAME"])
    interchange_tree = BallTree(np.radians(interchanges[["LATITUDE", "LONGITUDE"]].values), metric="haversine")

    dist_ic, _ = interchange_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_interchange_mrt"] = dist_ic[:, 0] * EARTH_RADIUS_KM
    spatial["interchanges_within_1km"] = interchange_tree.query_radius(block_coords_rad, r=1.0 / EARTH_RADIUS_KM, count_only=True)

    # 4. Shopping Malls
    malls = pd.read_csv(config.SHOPPING_MALLS_PATH).dropna(subset=["LATITUDE", "LONGITUDE"])
    malls_coords_rad = np.radians(malls[["LATITUDE", "LONGITUDE"]].values)
    malls_tree = BallTree(malls_coords_rad, metric="haversine")

    dist_mall, _ = malls_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_mall"] = dist_mall[:, 0] * EARTH_RADIUS_KM
    spatial["malls_within_1km"] = malls_tree.query_radius(block_coords_rad, r=1.0 / EARTH_RADIUS_KM, count_only=True)
    spatial["malls_within_2km"] = malls_tree.query_radius(block_coords_rad, r=2.0 / EARTH_RADIUS_KM, count_only=True)

    # 5. Schools & Top-Tier Schools (NEW)
    schools = pd.read_csv(config.SCHOOLS_PATH)
    is_primary = schools["MAINLEVEL_CODE"].astype(str).str.lower().str.contains("primary")
    primary_schools = schools[is_primary]
    primary_tree = BallTree(np.radians(primary_schools[["LATITUDE", "LONGITUDE"]].values), metric="haversine")

    dist_pri, _ = primary_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_primary_school"] = dist_pri[:, 0] * EARTH_RADIUS_KM
    spatial["primary_schools_within_1km"] = primary_tree.query_radius(block_coords_rad, r=1.0 / EARTH_RADIUS_KM, count_only=True)

    is_secondary = schools["MAINLEVEL_CODE"].astype(str).str.lower().str.contains("secondary")
    sec_schools = schools[is_secondary]
    sec_tree = BallTree(np.radians(sec_schools[["LATITUDE", "LONGITUDE"]].values), metric="haversine")
    dist_sec, _ = sec_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_secondary_school"] = dist_sec[:, 0] * EARTH_RADIUS_KM

    # Top-tier schools (independent or government-aided)
    is_top = schools["TYPE_CODE"].isin(["government-aided sch", "independent school", "specialised independent school"])
    top_schools = schools[is_top]
    top_tree = BallTree(np.radians(top_schools[["LATITUDE", "LONGITUDE"]].values), metric="haversine")
    dist_top, _ = top_tree.query(block_coords_rad, k=1)
    spatial["dist_nearest_top_school"] = dist_top[:, 0] * EARTH_RADIUS_KM
    spatial["top_schools_within_1km"] = top_tree.query_radius(block_coords_rad, r=1.0 / EARTH_RADIUS_KM, count_only=True)

    # Proximity inverse scores
    spatial["mrt_proximity_score"] = 1.0 / (spatial["dist_nearest_mrt"] + 0.1)
    spatial["mall_proximity_score"] = 1.0 / (spatial["dist_nearest_mall"] + 0.1)
    spatial["school_proximity_score"] = 1.0 / (spatial["dist_nearest_primary_school"] + 0.1)
    spatial["interchange_proximity_score"] = 1.0 / (spatial["dist_nearest_interchange_mrt"] + 0.1)

    print(f"   [Spatial FE v2] Completed {len(spatial.columns) - 6} spatial features for {len(spatial)} blocks.")
    return spatial.drop(columns=["LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"])

def build_median_rent_benchmark_v2() -> pd.DataFrame:
    """
    Builds quarterly HDB median rent benchmarks with Dynamic Growth Extrapolation for 2026-Q3.
    """
    print("   [Macro FE v2] Building quarterly HDB median rent benchmarks with trend extrapolation...")
    mr = pd.read_csv(config.HDB_MEDIAN_RENT_PATH)
    mr["town_clean"] = mr["town"].astype(str).str.strip().str.lower()
    mr["flat_type_clean"] = mr["flat_type"].map({
        "1-RM": "1-room", "2-RM": "2-room", "3-RM": "3-room",
        "4-RM": "4-room", "5-RM": "5-room", "EXEC": "executive"
    })
    mr["median_rent_num"] = pd.to_numeric(mr["median_rent"], errors="coerce")

    # National median by flat type
    nat_mr = mr.groupby(["quarter", "flat_type_clean"])["median_rent_num"].median().reset_index()
    nat_mr.rename(columns={"median_rent_num": "nat_median_rent_benchmark"}, inplace=True)

    nat_pivot = nat_mr.pivot(index="quarter", columns="flat_type_clean", values="nat_median_rent_benchmark")
    if "1-room" not in nat_pivot.columns or nat_pivot["1-room"].isnull().all():
        nat_pivot["1-room"] = nat_pivot["2-room"] * 0.65
    else:
        nat_pivot["1-room"] = nat_pivot["1-room"].fillna(nat_pivot["2-room"] * 0.65)
    nat_pivot["2-room"] = nat_pivot["2-room"].ffill().bfill()
    nat_pivot["1-room"] = nat_pivot["1-room"].fillna(nat_pivot["2-room"] * 0.65)

    nat_mr_filled = nat_pivot.reset_index().melt(id_vars="quarter", value_name="nat_median_rent_benchmark")
    full_mr = mr.merge(nat_mr_filled, on=["quarter", "flat_type_clean"], how="right")
    full_mr["median_rent_benchmark"] = full_mr["median_rent_num"].combine_first(full_mr["nat_median_rent_benchmark"])

    # Dynamic Extrapolation for 2026-Q3
    # Compute town/type growth from 2026-Q1 to 2026-Q2
    q1_2026 = full_mr[full_mr["quarter"] == "2026-Q1"].set_index(["town_clean", "flat_type_clean"])["median_rent_benchmark"]
    q2_2026 = full_mr[full_mr["quarter"] == "2026-Q2"].copy()
    
    # Calculate delta, clipping negative shocks to maintain conservative inflation
    delta = (q2_2026.set_index(["town_clean", "flat_type_clean"])["median_rent_benchmark"] - q1_2026).fillna(25.0)
    delta_clipped = delta.clip(lower=0.0, upper=60.0) # bounded reasonable quarterly appreciation

    q3_2026 = q2_2026.copy()
    q3_2026["quarter"] = "2026-Q3"
    q3_keys = q3_2026.set_index(["town_clean", "flat_type_clean"]).index
    q3_2026["median_rent_benchmark"] = q3_2026["median_rent_benchmark"].values + delta_clipped.reindex(q3_keys).fillna(25.0).values
    q3_2026["nat_median_rent_benchmark"] = q3_2026["nat_median_rent_benchmark"] + 25.0

    full_mr = pd.concat([full_mr, q3_2026], ignore_index=True)

    keep_cols = ["quarter", "town_clean", "flat_type_clean", "median_rent_benchmark", "nat_median_rent_benchmark"]
    clean_mr = full_mr[keep_cols].drop_duplicates(subset=["quarter", "town_clean", "flat_type_clean"])
    clean_mr["median_rent_ratio"] = clean_mr["median_rent_benchmark"] / clean_mr["nat_median_rent_benchmark"]
    return clean_mr

def build_coe_features_v2() -> pd.DataFrame:
    """Extract monthly COE features."""
    coe = pd.read_csv(config.COE_PRICES_PATH)
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    coe["approval_month"] = coe["MONTH"].astype(str).str.strip().str.lower().map(month_map)
    coe["approval_year"] = coe["YEAR"].astype(int)
    coe["quota_num"] = coe["QUOTA_PREMIUM"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    
    coe_pvt = coe.groupby(["approval_year", "approval_month", "CATEGORY"])["quota_num"].mean().unstack()
    coe_pvt.columns = [f"coe_{c.strip().lower().replace(' ', '_')}" for c in coe_pvt.columns]
    
    all_months = pd.MultiIndex.from_product([range(2021, 2027), range(1, 13)], names=["approval_year", "approval_month"])
    coe_full = coe_pvt.reindex(all_months).ffill().bfill().reset_index()
    coe_full["coe_overall_mean"] = coe_full[[c for c in coe_full.columns if c.startswith("coe_")]].mean(axis=1)
    return coe_full

def build_stock_features_v2() -> pd.DataFrame:
    """Extract Singapore banking composite stock index."""
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

    bank_df["sg_financial_index"] = (
        bank_df["stock_dbs_close"] +
        bank_df["stock_ocbc_close"] * 2.0 +
        bank_df["stock_uob_close"] +
        bank_df["stock_singtel_close"] * 5.0
    ) / 4.0

    all_months = pd.MultiIndex.from_product([range(2021, 2027), range(1, 13)], names=["approval_year", "approval_month"])
    stocks_full = bank_df.set_index(["approval_year", "approval_month"]).reindex(all_months).ffill().bfill().reset_index()
    return stocks_full[["approval_year", "approval_month", "stock_dbs_close", "stock_ocbc_close", "stock_uob_close", "sg_financial_index"]]

def compute_hierarchical_te(train_df: pd.DataFrame, target_col: str = "target_res", k_folds: int = 5):
    """
    Computes 5-Fold Out-of-Fold (OOF) Hierarchical Bayesian Target Encoding for train,
    and returns full-sample priors for mapping onto the test set.
    """
    print(f"   [Target Encoding v2] Computing {k_folds}-Fold OOF Hierarchical Bayesian Target Encodings...")
    
    # Identifiers
    train_df = train_df.copy()
    train_df["block_id"] = train_df["town_clean"] + "_" + train_df["block_clean"] + "_" + train_df["street_clean"]
    train_df["block_ft_id"] = train_df["block_id"] + "_" + train_df["flat_type_clean"]
    
    global_mean = train_df[target_col].mean()
    te_cols = ["te_town", "te_block", "te_bft", "te_street", "te_subzone"]
    
    # Function to calculate maps given training fold
    def get_shrinkage_maps(tr):
        # 1. Town
        t_agg = tr.groupby("town_clean")[target_col].agg(["count", "mean"])
        t_map = ((t_agg["count"] * t_agg["mean"] + 30.0 * global_mean) / (t_agg["count"] + 30.0)).to_dict()
        
        # 2. Block
        b_agg = tr.groupby("block_id").agg(count=(target_col, "count"), mean=(target_col, "mean"), town=("town_clean", "first"))
        b_agg["town_prior"] = b_agg["town"].map(t_map).fillna(global_mean)
        b_map = ((b_agg["count"] * b_agg["mean"] + 10.0 * b_agg["town_prior"]) / (b_agg["count"] + 10.0)).to_dict()
        
        # 3. Block x Flat Type
        bft_agg = tr.groupby("block_ft_id").agg(count=(target_col, "count"), mean=(target_col, "mean"), block_id=("block_id", "first"))
        bft_agg["block_prior"] = bft_agg["block_id"].map(b_map).fillna(global_mean)
        bft_map = ((bft_agg["count"] * bft_agg["mean"] + 5.0 * bft_agg["block_prior"]) / (bft_agg["count"] + 5.0)).to_dict()
        
        # 4. Street
        s_agg = tr.groupby("street_clean").agg(count=(target_col, "count"), mean=(target_col, "mean"), town=("town_clean", "first"))
        s_agg["town_prior"] = s_agg["town"].map(t_map).fillna(global_mean)
        s_map = ((s_agg["count"] * s_agg["mean"] + 15.0 * s_agg["town_prior"]) / (s_agg["count"] + 15.0)).to_dict()
        
        # 5. Subzone
        sub_agg = tr.groupby("subzone_clean")[target_col].agg(["count", "mean"])
        sub_map = ((sub_agg["count"] * sub_agg["mean"] + 20.0 * global_mean) / (sub_agg["count"] + 20.0)).to_dict()
        
        return t_map, b_map, bft_map, s_map, sub_map

    # K-Fold Out-of-Fold generation for training set (Strictly Zero Leakage)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.RANDOM_SEED)
    oof_te = pd.DataFrame(index=train_df.index, columns=te_cols, dtype=float)
    
    for tr_idx, va_idx in kf.split(train_df):
        f_tr = train_df.iloc[tr_idx]
        f_va = train_df.iloc[va_idx]
        
        t_map, b_map, bft_map, s_map, sub_map = get_shrinkage_maps(f_tr)
        
        va_te_town = f_va["town_clean"].map(t_map).fillna(global_mean)
        va_te_block = f_va["block_id"].map(b_map).fillna(va_te_town)
        va_te_bft = f_va["block_ft_id"].map(bft_map).fillna(va_te_block)
        va_te_street = f_va["street_clean"].map(s_map).fillna(va_te_town)
        va_te_subzone = f_va["subzone_clean"].map(sub_map).fillna(global_mean)
        
        oof_te.iloc[va_idx, 0] = va_te_town
        oof_te.iloc[va_idx, 1] = va_te_block
        oof_te.iloc[va_idx, 2] = va_te_bft
        oof_te.iloc[va_idx, 3] = va_te_street
        oof_te.iloc[va_idx, 4] = va_te_subzone

    # Full training maps for test set
    full_maps = get_shrinkage_maps(train_df)
    return oof_te, full_maps, global_mean

def apply_hierarchical_te(test_df: pd.DataFrame, full_maps, global_mean) -> pd.DataFrame:
    """Applies global historical Bayesian target encoding maps to test set."""
    t_map, b_map, bft_map, s_map, sub_map = full_maps
    
    test_df = test_df.copy()
    test_df["block_id"] = test_df["town_clean"] + "_" + test_df["block_clean"] + "_" + test_df["street_clean"]
    test_df["block_ft_id"] = test_df["block_id"] + "_" + test_df["flat_type_clean"]
    
    te_town = test_df["town_clean"].map(t_map).fillna(global_mean)
    te_block = test_df["block_id"].map(b_map).fillna(te_town)
    te_bft = test_df["block_ft_id"].map(bft_map).fillna(te_block)
    te_street = test_df["street_clean"].map(s_map).fillna(te_town)
    te_subzone = test_df["subzone_clean"].map(sub_map).fillna(global_mean)
    
    test_te = pd.DataFrame({
        "te_town": te_town,
        "te_block": te_block,
        "te_bft": te_bft,
        "te_street": te_street,
        "te_subzone": te_subzone
    }, index=test_df.index)
    return test_te

def build_feature_tables(train: pd.DataFrame, test: pd.DataFrame):
    """Build the V2 feature matrices in memory from the cleaned train/test frames (published median-rent benchmark)."""
    train, test = train.copy(), test.copy()
    
    # Generate canonical block_id identifier
    train["block_id"] = train["town_clean"] + "_" + train["block_clean"] + "_" + train["street_clean"]
    test["block_id"] = test["town_clean"] + "_" + test["block_clean"] + "_" + test["street_clean"]

    # 1. Physical Structural Ratios
    for df in [train, test]:
        df["floor_area_per_room"] = df["FLOOR_AREA_SQM"] / df["room_count"]
        df["floor_area_per_max_floor"] = df["FLOOR_AREA_SQM"] / df["MAX_FLOOR"]
        df["flat_age_squared"] = df["flat_age_at_approval"] ** 2

    # 2. Spatial BallTree Features
    blocks = pd.concat([
        train[["block_id", "LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"]],
        test[["block_id", "LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"]]
    ]).drop_duplicates(subset=["block_id"])
    
    spatial_features = build_spatial_features_v2(blocks)
    train = train.merge(spatial_features, on="block_id", how="left")
    test = test.merge(spatial_features, on="block_id", how="left")

    # 3. Macroeconomic Features
    median_rent = build_median_rent_benchmark_v2()
    train = train.merge(median_rent, left_on=["approval_quarter", "town_clean", "flat_type_clean"],
                        right_on=["quarter", "town_clean", "flat_type_clean"], how="left").drop(columns=["quarter"])
    test = test.merge(median_rent, left_on=["approval_quarter", "town_clean", "flat_type_clean"],
                       right_on=["quarter", "town_clean", "flat_type_clean"], how="left").drop(columns=["quarter"])

    coe = build_coe_features_v2()
    train = train.merge(coe, on=["approval_year", "approval_month"], how="left")
    test = test.merge(coe, on=["approval_year", "approval_month"], how="left")

    stocks = build_stock_features_v2()
    train = train.merge(stocks, on=["approval_year", "approval_month"], how="left")
    test = test.merge(stocks, on=["approval_year", "approval_month"], how="left")

    # 4. Hierarchical Bayesian Target Encodings
    train["target_res"] = train["MONTHLY_RENT"] - train["median_rent_benchmark"]
    oof_train_te, full_maps, global_mean = compute_hierarchical_te(train, target_col="target_res", k_folds=config.N_FOLDS)
    test_te = apply_hierarchical_te(test, full_maps, global_mean)

    train = pd.concat([train, oof_train_te], axis=1)
    test = pd.concat([test, test_te], axis=1)

    # Clean up temporary target
    train.drop(columns=["target_res"], inplace=True)

    # 5. Null Check & Integrity Audit
    train_nulls = train.isnull().sum().sum()
    test_nulls = test.isnull().sum().sum()
    print(f"\n[Feature Integrity Audit v2]:")
    print(f"Train Shape: {train.shape} | Nulls: {train_nulls}")
    print(f"Test Shape:  {test.shape} | Nulls: {test_nulls}")
    assert train_nulls == 0, "Train feature matrix contains unexpected nulls!"
    assert test_nulls == 0, "Test feature matrix contains unexpected nulls!"

    return train, test
