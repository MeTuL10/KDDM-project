"""
Data Cleaning and Harmonization Pipeline (Phase 2).
Handles:
1. Normalization of FLAT_TYPE, FLAT_MODEL, TOWN, BLOCK, STREET
2. Elimination of duplicate category representations (e.g., '1 room' vs '1-room')
3. Sentinel detection and remediation (e.g. -1 in MAX_FLOOR and YEAR_COMPLETED)
4. Elimination of zero-variance constant features (FURNISHED, FEE)
5. Parsing and engineering of temporal, age, and lease lifecycle features
6. Exact geospatial join with sg-hdb-block.csv
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional

try:
    from . import config
except ImportError:
    import config

def clean_string_series(series: pd.Series) -> pd.Series:
    """Strip whitespace, collapse internal multiple spaces, and convert to lowercase."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

def normalize_flat_type(series: pd.Series) -> pd.Series:
    """
    Standardize FLAT_TYPE values across train and test.
    Maps unhyphenated forms ('1 room', '4 room') to canonical hyphenated ('1-room', '4-room').
    """
    cleaned = clean_string_series(series)
    normalized = cleaned.str.replace(" room", "-room", regex=False)
    
    # Validation assertion
    valid_types = set(config.FLAT_TYPE_ORDER.keys())
    unique_found = set(normalized.unique())
    diff = unique_found - valid_types
    if diff:
        raise ValueError(f"Encountered unexpected FLAT_TYPE values: {diff}")
        
    return normalized

def normalize_flat_model(series: pd.Series) -> pd.Series:
    """
    Standardize FLAT_MODEL values.
    Standardizes '2 room' to '2-room' and maps singular rare models ('improved maisonette' -> 'maisonette').
    """
    cleaned = clean_string_series(series)
    # Apply synonym replacements
    for src, dst in config.MODEL_SYNONYMS.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned

def clean_block_data(block_path: str = config.HDB_BLOCK_PATH) -> pd.DataFrame:
    """
    Load and clean auxiliary sg-hdb-block.csv.
    Remediates -1 sentinel values in MAX_FLOOR and YEAR_COMPLETED.
    """
    df = pd.read_csv(block_path)
    
    # Normalize join keys
    df["town_clean"] = clean_string_series(df["TOWN"])
    df["block_clean"] = clean_string_series(df["BLOCK"])
    df["street_clean"] = clean_string_series(df["STREET"])
    
    # Standardize string fields
    df["subzone_clean"] = clean_string_series(df["SUBZONE"])
    df["planning_area_clean"] = clean_string_series(df["PLANNING_AREA"])
    df["region_clean"] = clean_string_series(df["REGION"])
    
    # Handle postal code NIL sentinel
    df["postal_code_clean"] = df["POSTAL_CODE"].astype(str).str.strip().str.upper()
    df.loc[df["postal_code_clean"].isin(["NIL", "NAN", "NA", ""]), "postal_code_clean"] = "UNKNOWN"

    # Identify -1 sentinel rows in MAX_FLOOR and YEAR_COMPLETED
    sentinel_mask = (df["MAX_FLOOR"] == -1) | (df["YEAR_COMPLETED"] == -1)
    
    # Compute robust fallback medians by street and town
    street_medians = df[~sentinel_mask].groupby("street_clean")[["MAX_FLOOR", "YEAR_COMPLETED"]].median()
    town_medians = df[~sentinel_mask].groupby("town_clean")[["MAX_FLOOR", "YEAR_COMPLETED"]].median()

    # Impute MAX_FLOOR sentinels
    for idx in df[df["MAX_FLOOR"] == -1].index:
        st = df.loc[idx, "street_clean"]
        twn = df.loc[idx, "town_clean"]
        if st in street_medians.index and not np.isnan(street_medians.loc[st, "MAX_FLOOR"]):
            df.loc[idx, "MAX_FLOOR"] = street_medians.loc[st, "MAX_FLOOR"]
        else:
            df.loc[idx, "MAX_FLOOR"] = town_medians.loc[twn, "MAX_FLOOR"]

    # Impute YEAR_COMPLETED sentinels (if not merged with lease commence date)
    for idx in df[df["YEAR_COMPLETED"] == -1].index:
        st = df.loc[idx, "street_clean"]
        twn = df.loc[idx, "town_clean"]
        if st in street_medians.index and not np.isnan(street_medians.loc[st, "YEAR_COMPLETED"]):
            df.loc[idx, "YEAR_COMPLETED"] = street_medians.loc[st, "YEAR_COMPLETED"]
        else:
            df.loc[idx, "YEAR_COMPLETED"] = town_medians.loc[twn, "YEAR_COMPLETED"]

    keep_cols = [
        "town_clean", "block_clean", "street_clean",
        "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
        "subzone_clean", "planning_area_clean", "region_clean", "postal_code_clean"
    ]
    return df[keep_cols].drop_duplicates(subset=["town_clean", "block_clean", "street_clean"])

def clean_dataset(
    df: pd.DataFrame,
    is_train: bool = True,
    block_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Comprehensive cleaning and initial feature extraction for Train or Test data.
    """
    cleaned = df.copy()

    # 1. Normalize Categorical Features
    cleaned["town_clean"] = clean_string_series(cleaned["TOWN"])
    cleaned["block_clean"] = clean_string_series(cleaned["BLOCK"])
    cleaned["street_clean"] = clean_string_series(cleaned["STREET"])
    cleaned["flat_type_clean"] = normalize_flat_type(cleaned["FLAT_TYPE"])
    cleaned["flat_model_clean"] = normalize_flat_model(cleaned["FLAT_MODEL"])
    
    # Ordinal numeric mapping for flat size
    cleaned["room_count"] = cleaned["flat_type_clean"].map(config.FLAT_TYPE_ORDER).astype(int)

    # 2. Drop Zero-Variance Constant Attributes
    drop_const_cols = [c for c in ["FURNISHED", "FEE"] if c in cleaned.columns]
    if drop_const_cols:
        cleaned = cleaned.drop(columns=drop_const_cols)

    # 3. Parse Temporal Signals
    cleaned["approval_year"] = cleaned["RENT_APPROVAL_DATE"].str[:4].astype(int)
    cleaned["approval_month"] = cleaned["RENT_APPROVAL_DATE"].str[5:7].astype(int)
    cleaned["approval_quarter_num"] = (cleaned["approval_month"] - 1) // 3 + 1
    cleaned["approval_quarter"] = (
        cleaned["approval_year"].astype(str) + "-Q" + cleaned["approval_quarter_num"].astype(str)
    )
    
    # Continuous linear time step (0 = Jan 2021)
    cleaned["time_step"] = (cleaned["approval_year"] - 2021) * 12 + (cleaned["approval_month"] - 1)
    
    # Cyclical seasonality transforms
    cleaned["month_sin"] = np.sin(2 * np.pi * cleaned["approval_month"] / 12.0)
    cleaned["month_cos"] = np.cos(2 * np.pi * cleaned["approval_month"] / 12.0)

    # 4. Lease & Flat Age Lifecycle Signals
    cleaned["lease_commence_year"] = cleaned["LEASE_COMMENCE_DATE"].astype(int)
    cleaned["flat_age_at_approval"] = cleaned["approval_year"] - cleaned["lease_commence_year"]
    cleaned["remaining_lease_years"] = 99 - cleaned["flat_age_at_approval"]

    # 5. Exact Geospatial Join with Auxiliary HDB Block Data
    if block_df is None:
        block_df = clean_block_data()

    merged = cleaned.merge(
        block_df,
        on=["town_clean", "block_clean", "street_clean"],
        how="left"
    )

    # Sanity verification: Ensure 100% of rows match
    unmatched_count = merged["LATITUDE"].isnull().sum()
    if unmatched_count > 0:
        raise RuntimeError(f"Found {unmatched_count} unmatched address rows after geospatial join!")

    # For any block where year_completed was sentinel -1, fallback to lease_commence_year if closer
    merged["year_completed_clean"] = merged["YEAR_COMPLETED"]
    sentinel_comp_mask = merged["year_completed_clean"] <= 0
    merged.loc[sentinel_comp_mask, "year_completed_clean"] = merged.loc[sentinel_comp_mask, "lease_commence_year"]
    
    # Calculate difference between lease commencement and building completion
    merged["lease_lag_years"] = merged["lease_commence_year"] - merged["year_completed_clean"]

    # High rise building flag
    merged["is_high_rise"] = (merged["MAX_FLOOR"] >= 20).astype(int)

    # Final cleanup of raw redundant string columns (retaining clean standardized ones)
    raw_to_drop = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "LEASE_COMMENCE_DATE", "YEAR_COMPLETED"]
    merged = merged.drop(columns=[c for c in raw_to_drop if c in merged.columns])

    return merged

def prepare_cleaned_datasets(
    drop_train_exact_duplicates: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point to execute Phase 2 pipeline.
    Loads raw train/test, executes cleaning, and returns processed DataFrames.
    """
    print("=" * 80)
    print("PHASE 2: EXECUTING DATA CLEANING & PREPROCESSING PIPELINE")
    print("=" * 80)

    # Ensure output directory exists
    os.makedirs(config.CLEANED_DATA_DIR, exist_ok=True)

    print("1. Loading raw datasets and auxiliary block data...")
    raw_train = pd.read_csv(config.TRAIN_PATH)
    raw_test = pd.read_csv(config.TEST_PATH)
    block_df = clean_block_data()
    print(f"   Raw Train: {raw_train.shape}, Raw Test: {raw_test.shape}, Cleaned Blocks: {block_df.shape}")

    # Check and log exact duplicates in train
    exact_dups = raw_train.duplicated().sum()
    print(f"2. Exact duplicate count in Train: {exact_dups}")
    if drop_train_exact_duplicates:
        print("   -> Dropping exact duplicate rows from Train as requested.")
        raw_train = raw_train.drop_duplicates()
        print(f"   -> Train shape after dropping exact duplicates: {raw_train.shape}")
    else:
        print("   -> Retaining exact duplicate rows (representing legitimate multiple concurrent tenancies).")

    print("3. Cleaning Train dataset...")
    clean_train = clean_dataset(raw_train, is_train=True, block_df=block_df)
    
    print("4. Cleaning Test dataset...")
    clean_test = clean_dataset(raw_test, is_train=False, block_df=block_df)

    # Verification assertions
    assert len(clean_test) == 50000, f"Test row count changed: {len(clean_test)} != 50000"
    assert clean_train.isnull().sum().sum() == 0, f"Found unexpected nulls in clean_train:\n{clean_train.isnull().sum()[clean_train.isnull().sum() > 0]}"
    assert clean_test.isnull().sum().sum() == 0, f"Found unexpected nulls in clean_test:\n{clean_test.isnull().sum()[clean_test.isnull().sum() > 0]}"

    print("5. Saving cleaned datasets to disk...")
    train_out_path = os.path.join(config.CLEANED_DATA_DIR, "train_cleaned.parquet")
    test_out_path = os.path.join(config.CLEANED_DATA_DIR, "test_cleaned.parquet")
    
    clean_train.to_parquet(train_out_path, index=False)
    clean_test.to_parquet(test_out_path, index=False)
    print(f"   Saved cleaned train: {train_out_path} ({clean_train.shape})")
    print(f"   Saved cleaned test:  {test_out_path} ({clean_test.shape})")

    print("\nSummary of Cleaned Train Columns & Data Types:")
    for col, dtype in clean_train.dtypes.items():
        print(f"   - {col:25s}: {str(dtype):12s} (sample: {clean_train[col].iloc[0]})")

    print("\nData Cleaning & Preprocessing (Phase 2) completed successfully!")
    return clean_train, clean_test

if __name__ == "__main__":
    prepare_cleaned_datasets()
