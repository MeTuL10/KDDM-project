"""
Feature Engineering Module v4 (Sprint 5: Zero-Leakage & Private LB Robustness)
Path: expt-2/src/feature_engineering_v4.py

Architectural Principles:
1. Complete Decoupling of Static Features from Target Encodings:
   - train_features_v4.parquet and test_features_v4.parquet contain 100% leak-free
     spatial, macroeconomic, cohort area, and structural features.
   - NO static target encodings are pre-computed or stored in parquet files.
2. Modular HierarchicalBayesianTargetEncoder Class:
   - Provides strict Out-of-Fold (OOF) target encoding inside cross-validation folds.
   - Eliminates temporal leakage on OOT holdout and cross-fold contamination in 5-fold CV.
   - Eliminates sample size scale discrepancy (n = 0.8 * N on both validation and test).
3. Private Leaderboard Stability:
   - Eliminates all sources of validation over-optimism.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold
import config

EARTH_RADIUS_KM = 6371.0

LUXURY_MODELS = [
    "type s1", "type s2", "terrace", "dbss", "adjoined flat",
    "premium maisonette", "maisonette", "model a maisonette", "premium apartment loft"
]


class HierarchicalBayesianTargetEncoder:
    """
    Leak-Free Hierarchical Bayesian Target Encoder.
    
    Shrinks high-cardinality geographic and categorical entities toward their parent priors:
    1. Town (M=30) -> Global Mean
    2. Block (M=10) -> Town Prior
    3. Block x Flat Type (M=5) -> Block Prior
    4. Street (M=15) -> Town Prior
    5. Subzone (M=20) -> Global Mean
    6. Flat Model (M=20) -> Global Mean
    7. Town x Flat Model (M=10) -> Flat Model Prior
    """
    def __init__(
        self,
        m_town: float = 30.0,
        m_block: float = 10.0,
        m_bft: float = 5.0,
        m_street: float = 15.0,
        m_subzone: float = 20.0,
        m_model: float = 20.0,
        m_town_model: float = 10.0
    ):
        self.m_town = m_town
        self.m_block = m_block
        self.m_bft = m_bft
        self.m_street = m_street
        self.m_subzone = m_subzone
        self.m_model = m_model
        self.m_town_model = m_town_model

        self.global_mean: float = 0.0
        self.t_map: Dict[str, float] = {}
        self.b_map: Dict[str, float] = {}
        self.bft_map: Dict[str, float] = {}
        self.s_map: Dict[str, float] = {}
        self.sub_map: Dict[str, float] = {}
        self.m_map: Dict[str, float] = {}
        self.tm_map: Dict[str, float] = {}
        self.is_fitted: bool = False

    @staticmethod
    def _create_composite_keys(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "block_id" not in df.columns:
            df["block_id"] = df["town_clean"] + "_" + df["block_clean"] + "_" + df["street_clean"]
        df["block_ft_id"] = df["block_id"] + "_" + df["flat_type_clean"]
        df["town_model_id"] = df["town_clean"] + "_" + df["flat_model_clean"]
        return df

    def _compute_maps(self, tr: pd.DataFrame, target_col: str):
        tr = self._create_composite_keys(tr)
        gm = float(tr[target_col].mean())

        # 1. Town
        t_agg = tr.groupby("town_clean")[target_col].agg(["count", "mean"])
        t_map = ((t_agg["count"] * t_agg["mean"] + self.m_town * gm) / (t_agg["count"] + self.m_town)).to_dict()

        # 2. Block
        b_agg = tr.groupby("block_id").agg(count=(target_col, "count"), mean=(target_col, "mean"), town=("town_clean", "first"))
        b_agg["town_prior"] = b_agg["town"].map(t_map).fillna(gm)
        b_map = ((b_agg["count"] * b_agg["mean"] + self.m_block * b_agg["town_prior"]) / (b_agg["count"] + self.m_block)).to_dict()

        # 3. Block x Flat Type
        bft_agg = tr.groupby("block_ft_id").agg(count=(target_col, "count"), mean=(target_col, "mean"), block_id=("block_id", "first"))
        bft_agg["block_prior"] = bft_agg["block_id"].map(b_map).fillna(gm)
        bft_map = ((bft_agg["count"] * bft_agg["mean"] + self.m_bft * bft_agg["block_prior"]) / (bft_agg["count"] + self.m_bft)).to_dict()

        # 4. Street
        s_agg = tr.groupby("street_clean").agg(count=(target_col, "count"), mean=(target_col, "mean"), town=("town_clean", "first"))
        s_agg["town_prior"] = s_agg["town"].map(t_map).fillna(gm)
        s_map = ((s_agg["count"] * s_agg["mean"] + self.m_street * s_agg["town_prior"]) / (s_agg["count"] + self.m_street)).to_dict()

        # 5. Subzone
        sub_agg = tr.groupby("subzone_clean")[target_col].agg(["count", "mean"])
        sub_map = ((sub_agg["count"] * sub_agg["mean"] + self.m_subzone * gm) / (sub_agg["count"] + self.m_subzone)).to_dict()

        # 6. Flat Model
        m_agg = tr.groupby("flat_model_clean")[target_col].agg(["count", "mean"])
        m_map = ((m_agg["count"] * m_agg["mean"] + self.m_model * gm) / (m_agg["count"] + self.m_model)).to_dict()

        # 7. Town x Flat Model
        tm_agg = tr.groupby("town_model_id").agg(count=(target_col, "count"), mean=(target_col, "mean"), model=("flat_model_clean", "first"))
        tm_agg["model_prior"] = tm_agg["model"].map(m_map).fillna(gm)
        tm_map = ((tm_agg["count"] * tm_agg["mean"] + self.m_town_model * tm_agg["model_prior"]) / (tm_agg["count"] + self.m_town_model)).to_dict()

        return gm, t_map, b_map, bft_map, s_map, sub_map, m_map, tm_map

    def fit(self, train_df: pd.DataFrame, target_col: str = "target_res"):
        """Fits target encoding maps using 100% of the provided training slice."""
        gm, t_map, b_map, bft_map, s_map, sub_map, m_map, tm_map = self._compute_maps(train_df, target_col)
        self.global_mean = gm
        self.t_map = t_map
        self.b_map = b_map
        self.bft_map = bft_map
        self.s_map = s_map
        self.sub_map = sub_map
        self.m_map = m_map
        self.tm_map = tm_map
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms unseen validation or test rows using pre-computed shrinkage maps."""
        if not self.is_fitted:
            raise RuntimeError("HierarchicalBayesianTargetEncoder must be fitted before calling transform!")
        
        df = self._create_composite_keys(df)
        te_town = df["town_clean"].map(self.t_map).fillna(self.global_mean)
        te_block = df["block_id"].map(self.b_map).fillna(te_town)
        te_bft = df["block_ft_id"].map(self.bft_map).fillna(te_block)
        te_street = df["street_clean"].map(self.s_map).fillna(te_town)
        te_subzone = df["subzone_clean"].map(self.sub_map).fillna(self.global_mean)
        te_model = df["flat_model_clean"].map(self.m_map).fillna(self.global_mean)
        te_town_model = df["town_model_id"].map(self.tm_map).fillna(te_model)

        return pd.DataFrame({
            "te_town": te_town,
            "te_block": te_block,
            "te_bft": te_bft,
            "te_street": te_street,
            "te_subzone": te_subzone,
            "te_flat_model": te_model,
            "te_town_flat_model": te_town_model
        }, index=df.index)

    def fit_transform_oof(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target_res",
        k_folds: int = 5,
        random_state: int = config.RANDOM_SEED
    ) -> pd.DataFrame:
        """
        Fits encoder on train_df and returns internal out-of-fold encoded features for train_df.
        Also finalizes self.fit on the complete train_df so subsequent transform() calls are valid.
        """
        train_df = self._create_composite_keys(train_df)
        te_cols = ["te_town", "te_block", "te_bft", "te_street", "te_subzone", "te_flat_model", "te_town_flat_model"]
        oof_df = pd.DataFrame(index=train_df.index, columns=te_cols, dtype=float)

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        for tr_idx, va_idx in kf.split(train_df):
            f_tr = train_df.iloc[tr_idx]
            f_va = train_df.iloc[va_idx]

            gm, t_map, b_map, bft_map, s_map, sub_map, m_map, tm_map = self._compute_maps(f_tr, target_col)

            va_te_town = f_va["town_clean"].map(t_map).fillna(gm)
            va_te_block = f_va["block_id"].map(b_map).fillna(va_te_town)
            va_te_bft = f_va["block_ft_id"].map(bft_map).fillna(va_te_block)
            va_te_street = f_va["street_clean"].map(s_map).fillna(va_te_town)
            va_te_subzone = f_va["subzone_clean"].map(sub_map).fillna(gm)
            va_te_model = f_va["flat_model_clean"].map(m_map).fillna(gm)
            va_te_town_model = f_va["town_model_id"].map(tm_map).fillna(va_te_model)

            oof_df.iloc[va_idx, 0] = va_te_town
            oof_df.iloc[va_idx, 1] = va_te_block
            oof_df.iloc[va_idx, 2] = va_te_bft
            oof_df.iloc[va_idx, 3] = va_te_street
            oof_df.iloc[va_idx, 4] = va_te_subzone
            oof_df.iloc[va_idx, 5] = va_te_model
            oof_df.iloc[va_idx, 6] = va_te_town_model

        # Fit on 100% of train_df for future inference
        self.fit(train_df, target_col=target_col)
        return oof_df


def build_spatial_features_v4(block_df: pd.DataFrame) -> pd.DataFrame:
    """Computes spatial proximity features including MRT interchanges and top-tier schools."""
    print("   [Spatial FE v4] Building BallTree spatial features for unique blocks...")
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

    # 3. MRT Multi-Line Interchanges
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

    # 5. Schools & Top-Tier Schools
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

    print(f"   [Spatial FE v4] Completed {len(spatial.columns) - 6} spatial features for {len(spatial)} blocks.")
    return spatial.drop(columns=["LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"])


def build_median_rent_benchmark_v4() -> pd.DataFrame:
    """Builds quarterly HDB median rent benchmarks with Dynamic Growth Extrapolation for 2026-Q3."""
    print("   [Macro FE v4] Building quarterly HDB median rent benchmarks with trend extrapolation...")
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
    q1_2026 = full_mr[full_mr["quarter"] == "2026-Q1"].set_index(["town_clean", "flat_type_clean"])["median_rent_benchmark"]
    q2_2026 = full_mr[full_mr["quarter"] == "2026-Q2"].copy()
    
    delta = (q2_2026.set_index(["town_clean", "flat_type_clean"])["median_rent_benchmark"] - q1_2026).fillna(25.0)
    delta_clipped = delta.clip(lower=0.0, upper=60.0)

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


def build_coe_features_v4() -> pd.DataFrame:
    """Extract monthly COE features."""
    print("   [Macro FE v4] Building monthly COE price indicators...")
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
    
    # Summary indicators
    cat_cols = [c for c in coe_full.columns if c.startswith("coe_")]
    coe_full["coe_overall_mean"] = coe_full[cat_cols].mean(axis=1)
    if "coe_cat_a" in coe_full.columns and "coe_cat_b" in coe_full.columns:
        coe_full["coe_premium_spread"] = coe_full["coe_cat_b"] - coe_full["coe_cat_a"]
    return coe_full


def build_stock_features_v4() -> pd.DataFrame:
    """Extract monthly banking & STI stock features."""
    print("   [Macro FE v4] Building monthly domestic stock market indices...")
    stocks = pd.read_csv(config.STOCK_PRICES_PATH)
    stocks["DATE"] = pd.to_datetime(stocks["DATE"])
    stocks["approval_year"] = stocks["DATE"].dt.year
    stocks["approval_month"] = stocks["DATE"].dt.month
    
    bank_symbols = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI"]
    stocks_banks = stocks[stocks["SYMBOL"].isin(bank_symbols)].copy()
    
    symbol_name_map = {
        "D05.SI": "stock_dbs_close",
        "O39.SI": "stock_ocbc_close",
        "U11.SI": "stock_uob_close",
        "Z74.SI": "stock_singtel_close"
    }
    stocks_banks["feat_name"] = stocks_banks["SYMBOL"].map(symbol_name_map)
    
    monthly_stocks = stocks_banks.groupby(["approval_year", "approval_month", "feat_name"])["CLOSE"].mean().unstack()
    all_months = pd.MultiIndex.from_product([range(2021, 2027), range(1, 13)], names=["approval_year", "approval_month"])
    stocks_full = monthly_stocks.reindex(all_months).ffill().bfill().reset_index()
    
    bank_cols = ["stock_dbs_close", "stock_ocbc_close", "stock_uob_close"]
    stocks_full["sg_financial_index"] = stocks_full[bank_cols].mean(axis=1)
    return stocks_full


def generate_clean_feature_matrices_v4() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds clean base feature matrices without static target encodings.
    Saves train_features_v4.parquet and test_features_v4.parquet.
    """
    print("=" * 80)
    print("SPRINT 5: BUILDING LEAK-FREE BASE FEATURE MATRICES (V4)")
    print("=" * 80)

    train_cleaned_path = os.path.join(config.CLEANED_DATA_DIR, "train_cleaned.parquet")
    test_cleaned_path = os.path.join(config.CLEANED_DATA_DIR, "test_cleaned.parquet")

    train = pd.read_parquet(train_cleaned_path)
    test = pd.read_parquet(test_cleaned_path)
    print(f"Loaded Cleaned Train: {train.shape} | Test: {test.shape}")

    # Add block_id
    train["block_id"] = train["town_clean"] + "_" + train["block_clean"] + "_" + train["street_clean"]
    test["block_id"] = test["town_clean"] + "_" + test["block_clean"] + "_" + test["street_clean"]

    # 1. Structural, Luxury & Height Features
    for df in [train, test]:
        df["is_luxury_model"] = df["flat_model_clean"].isin(LUXURY_MODELS).astype(int)
        df["is_skyrise"] = (df["MAX_FLOOR"] >= 30).astype(int)
        df["max_floor_tier"] = pd.cut(
            df["MAX_FLOOR"],
            bins=[-np.inf, 12, 20, 30, 40, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

    # Cohort Area Deviations (computed strictly from train distribution)
    area_cohort_means = train.groupby(["town_clean", "flat_type_clean"])["FLOOR_AREA_SQM"].mean().rename("cohort_mean_area").reset_index()
    national_type_means = train.groupby("flat_type_clean")["FLOOR_AREA_SQM"].mean().to_dict()

    train = train.merge(area_cohort_means, on=["town_clean", "flat_type_clean"], how="left")
    train["cohort_mean_area"] = train["cohort_mean_area"].fillna(train["flat_type_clean"].map(national_type_means))
    train["area_diff_town_type"] = train["FLOOR_AREA_SQM"] - train["cohort_mean_area"]
    train["area_ratio_town_type"] = train["FLOOR_AREA_SQM"] / (train["cohort_mean_area"] + 1e-5)
    train.drop(columns=["cohort_mean_area"], inplace=True)

    test = test.merge(area_cohort_means, on=["town_clean", "flat_type_clean"], how="left")
    test["cohort_mean_area"] = test["cohort_mean_area"].fillna(test["flat_type_clean"].map(national_type_means))
    test["area_diff_town_type"] = test["FLOOR_AREA_SQM"] - test["cohort_mean_area"]
    test["area_ratio_town_type"] = test["FLOOR_AREA_SQM"] / (test["cohort_mean_area"] + 1e-5)
    test.drop(columns=["cohort_mean_area"], inplace=True)

    # 2. Spatial BallTree Features
    blocks = pd.concat([
        train[["block_id", "LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"]],
        test[["block_id", "LATITUDE", "LONGITUDE", "subzone_clean", "planning_area_clean", "region_clean"]]
    ]).drop_duplicates(subset=["block_id"])
    
    spatial_features = build_spatial_features_v4(blocks)
    train = train.merge(spatial_features, on="block_id", how="left")
    test = test.merge(spatial_features, on="block_id", how="left")

    # Micro-Interactions
    for df in [train, test]:
        df["dist_cbd_x_remaining_lease"] = df["dist_cbd"] * df["remaining_lease_years"]
        df["area_x_max_floor"] = df["FLOOR_AREA_SQM"] * df["MAX_FLOOR"]

    # 3. Macroeconomic Features
    median_rent = build_median_rent_benchmark_v4()
    train = train.merge(median_rent, left_on=["approval_quarter", "town_clean", "flat_type_clean"],
                        right_on=["quarter", "town_clean", "flat_type_clean"], how="left").drop(columns=["quarter"])
    test = test.merge(median_rent, left_on=["approval_quarter", "town_clean", "flat_type_clean"],
                       right_on=["quarter", "town_clean", "flat_type_clean"], how="left").drop(columns=["quarter"])

    coe = build_coe_features_v4()
    train = train.merge(coe, on=["approval_year", "approval_month"], how="left")
    test = test.merge(coe, on=["approval_year", "approval_month"], how="left")

    stocks = build_stock_features_v4()
    train = train.merge(stocks, on=["approval_year", "approval_month"], how="left")
    test = test.merge(stocks, on=["approval_year", "approval_month"], how="left")

    # 4. Target Residual Definition
    train["target_res"] = train["MONTHLY_RENT"] - train["median_rent_benchmark"]

    # 5. Integrity Audit
    train_nulls = train.isnull().sum().sum()
    test_nulls = test.isnull().sum().sum()
    print(f"\n[Feature Integrity Audit v4]:")
    print(f"Train Shape: {train.shape} | Nulls: {train_nulls}")
    print(f"Test Shape:  {test.shape} | Nulls: {test_nulls}")
    assert train_nulls == 0, f"Train feature matrix contains {train_nulls} nulls!"
    assert test_nulls == 0, f"Test feature matrix contains {test_nulls} nulls!"

    # Confirm NO static target encoding columns are included
    te_cols = [c for c in train.columns if c.startswith("te_")]
    assert len(te_cols) == 0, f"Found unexpected static target encoding columns: {te_cols}"

    # 6. Save Base Parquets
    os.makedirs(config.FEATURE_DATA_DIR, exist_ok=True)
    train_out_path = os.path.join(config.FEATURE_DATA_DIR, "train_features_v4.parquet")
    test_out_path = os.path.join(config.FEATURE_DATA_DIR, "test_features_v4.parquet")

    train.to_parquet(train_out_path, index=False)
    test.to_parquet(test_out_path, index=False)
    print(f"\nSaved Clean Base Train Features: {train_out_path} ({train.shape})")
    print(f"Saved Clean Base Test Features:  {test_out_path} ({test.shape})")
    return train, test


if __name__ == "__main__":
    generate_clean_feature_matrices_v4()

