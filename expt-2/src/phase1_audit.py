"""
Phase 1: Exploratory Data Analysis & Comprehensive Data Quality Audit
Investigates:
1. Missing values and String/Numeric Sentinels
2. Exact duplicates and Feature-only duplicates
3. Categorical inconsistencies across attributes (FLAT_TYPE, FLAT_MODEL, TOWN, STREET, BLOCK)
4. Zero-variance attributes (FURNISHED, FEE)
5. Numerical distributions, skewness, and target MONTHLY_RENT characteristics
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
AUX_DIR = os.path.join(DATA_DIR, "auxiliary")

def audit_dataset():
    print("=" * 80)
    print("PHASE 1: DATA QUALITY & SENTINEL AUDIT")
    print("=" * 80)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train Shape: {train.shape}")
    print(f"Test Shape:  {test.shape}")
    print()

    # -------------------------------------------------------------
    # 1. MISSING VALUES & SENTINELS CHECK
    # -------------------------------------------------------------
    print(">>> 1. MISSING & SENTINEL VALUES AUDIT")
    sentinel_patterns = [
        r"^\s*$", r"^na$", r"^null$", r"^nil$", r"^none$", r"^nan$", r"^\?$", r"^-$", r"^--$", r"^n/a$"
    ]
    sentinel_regex = re.compile("|".join(sentinel_patterns), re.IGNORECASE)

    for name, df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n--- {name} Dataset ---")
        null_counts = df.isnull().sum()
        print("Standard Null Counts:")
        print(null_counts[null_counts > 0] if (null_counts > 0).any() else "No standard NaN/Null values.")

        print("Checking for String Sentinels ('na', 'null', 'nil', '-', '?', etc.):")
        sentinel_found = False
        for col in df.select_dtypes(include=["object"]).columns:
            matches = df[col].astype(str).str.strip().str.match(sentinel_regex)
            count = matches.sum()
            if count > 0:
                sentinel_found = True
                samples = df.loc[matches, col].unique()
                print(f"  Column '{col}': {count} sentinel values found. Samples: {samples}")
        if not sentinel_found:
            print("  No string sentinels found in any column.")

        print("Checking Numeric Columns for Suspicious Outliers / Sentinels (-1, 0, 999, 9999):")
        for col in df.select_dtypes(include=[np.number]).columns:
            zeros = (df[col] == 0).sum()
            negatives = (df[col] < 0).sum()
            extreme_999 = (df[col].isin([99, 999, 9999, 99999])).sum()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"  Column '{col}': min={min_val}, max={max_val}, zeros={zeros}, negatives={negatives}, sentinel_999s={extreme_999}")

    # Also audit auxiliary files for sentinels
    print("\n--- Auxiliary Datasets Sentinel Check ---")
    aux_files = sorted(os.listdir(AUX_DIR))
    for f in aux_files:
        if f.endswith(".csv"):
            aux_path = os.path.join(AUX_DIR, f)
            aux_df = pd.read_csv(aux_path, low_memory=False)
            sentinels_in_aux = {}
            for col in aux_df.columns:
                matches = aux_df[col].astype(str).str.strip().str.match(sentinel_regex)
                if matches.sum() > 0:
                    sentinels_in_aux[col] = (matches.sum(), aux_df.loc[matches, col].unique().tolist()[:3])
            if sentinels_in_aux:
                print(f"  File {f}: Sentinels found in:")
                for c, (cnt, smp) in sentinels_in_aux.items():
                    print(f"    - {c}: {cnt} rows, e.g. {smp}")
            else:
                print(f"  File {f}: Clean (no string sentinels detected).")

    # -------------------------------------------------------------
    # 2. DUPLICATE VALUES AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 2. DUPLICATE ANALYSIS")
    print("=" * 80)

    # Exact duplicates in Train (including MONTHLY_RENT)
    exact_train_dups = train.duplicated().sum()
    print(f"Train Exact Duplicate Rows (all columns identical): {exact_train_dups} ({exact_train_dups / len(train) * 100:.2f}%)")

    # Feature duplicates in Train (same features, but potentially different or same rent)
    feat_cols = [c for c in train.columns if c != "MONTHLY_RENT"]
    feature_train_dups = train.duplicated(subset=feat_cols, keep=False).sum()
    print(f"Train Rows with Identical Feature Tuples: {feature_train_dups} ({feature_train_dups / len(train) * 100:.2f}%)")

    # Look closer at feature duplicates: do they have different rents or identical rents?
    if exact_train_dups > 0:
        dup_sample = train[train.duplicated(subset=feat_cols, keep=False)].sort_values(by=feat_cols).head(10)
        print("\nSample of rows with identical features:")
        print(dup_sample[["RENT_APPROVAL_DATE", "TOWN", "BLOCK", "STREET", "FLAT_TYPE", "MONTHLY_RENT"]])

    # Exact duplicates in Test
    test_dups = test.duplicated().sum()
    print(f"\nTest Exact Duplicate Rows: {test_dups} ({test_dups / len(test) * 100:.2f}%)")

    # -------------------------------------------------------------
    # 3. CATEGORICAL INCONSISTENCIES & NORMALIZATION AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 3. CATEGORICAL VALUES & NORMALIZATION AUDIT")
    print("=" * 80)

    # FLAT_TYPE
    print("\n[FLAT_TYPE]")
    train_ft = train["FLAT_TYPE"].value_counts().to_dict()
    test_ft = test["FLAT_TYPE"].value_counts().to_dict()
    print("Train FLAT_TYPE counts:", train_ft)
    print("Test FLAT_TYPE counts: ", test_ft)

    # FLAT_MODEL
    print("\n[FLAT_MODEL]")
    print(f"Train unique FLAT_MODEL count: {train['FLAT_MODEL'].nunique()}")
    print(f"Test unique FLAT_MODEL count:  {test['FLAT_MODEL'].nunique()}")
    train_models = set(train["FLAT_MODEL"].str.strip().str.lower().unique())
    test_models = set(test["FLAT_MODEL"].str.strip().str.lower().unique())
    diff_models = test_models - train_models
    print("FLAT_MODEL in Test but not in Train (after lowercase):", diff_models if diff_models else "None (100% overlap)")
    print("Raw unique values sample:")
    print("Train raw sample:", train["FLAT_MODEL"].unique()[:10])
    print("Test raw sample: ", test["FLAT_MODEL"].unique()[:10])

    # TOWN
    print("\n[TOWN]")
    train_towns = set(train["TOWN"].str.strip().str.lower().unique())
    test_towns = set(test["TOWN"].str.strip().str.lower().unique())
    print(f"Train unique towns: {len(train_towns)}, Test unique towns: {len(test_towns)}")
    diff_towns = test_towns - train_towns
    print("Towns in Test but not in Train:", diff_towns if diff_towns else "None (100% overlap)")

    # STREET & BLOCK
    print("\n[STREET & BLOCK]")
    train["clean_block"] = train["BLOCK"].astype(str).str.strip().str.lower()
    train["clean_street"] = train["STREET"].astype(str).str.strip().str.lower()
    test["clean_block"] = test["BLOCK"].astype(str).str.strip().str.lower()
    test["clean_street"] = test["STREET"].astype(str).str.strip().str.lower()

    train_addrs = set(train["TOWN"].str.lower() + "||" + train["clean_block"] + "||" + train["clean_street"])
    test_addrs = set(test["TOWN"].str.lower() + "||" + test["clean_block"] + "||" + test["clean_street"])
    print(f"Unique Address Tuples in Train: {len(train_addrs)}")
    print(f"Unique Address Tuples in Test:  {len(test_addrs)}")
    print(f"Test addresses also in Train:   {len(test_addrs & train_addrs)} ({len(test_addrs & train_addrs)/len(test_addrs)*100:.2f}%)")
    print(f"Test addresses NEW (unseen):    {len(test_addrs - train_addrs)} ({len(test_addrs - train_addrs)/len(test_addrs)*100:.2f}%)")

    # Match with sg-hdb-block.csv
    hdb_blocks = pd.read_csv(os.path.join(AUX_DIR, "sg-hdb-block.csv"))
    hdb_blocks["clean_block"] = hdb_blocks["BLOCK"].astype(str).str.strip().str.lower()
    hdb_blocks["clean_street"] = hdb_blocks["STREET"].astype(str).str.strip().str.lower()
    hdb_blocks["clean_town"] = hdb_blocks["TOWN"].astype(str).str.strip().str.lower()
    hdb_addrs = set(hdb_blocks["clean_town"] + "||" + hdb_blocks["clean_block"] + "||" + hdb_blocks["clean_street"])

    unmatched_train = train_addrs - hdb_addrs
    unmatched_test = test_addrs - hdb_addrs
    print(f"Train addresses unmatched in sg-hdb-block: {len(unmatched_train)}")
    print(f"Test addresses unmatched in sg-hdb-block:  {len(unmatched_test)}")

    # -------------------------------------------------------------
    # 4. ZERO VARIANCE ATTRIBUTES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 4. ZERO-VARIANCE ATTRIBUTES AUDIT")
    print("=" * 80)
    for col in ["FURNISHED", "FEE"]:
        print(f"Train '{col}' unique values: {train[col].unique()}")
        print(f"Test  '{col}' unique values: {test[col].unique()}")

    # -------------------------------------------------------------
    # 5. TARGET VARIABLE (MONTHLY_RENT) DISTRIBUTION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> 5. TARGET VARIABLE (MONTHLY_RENT) AUDIT")
    print("=" * 80)
    rent = train["MONTHLY_RENT"]
    print(f"Count:    {rent.count()}")
    print(f"Mean:     {rent.mean():.2f}")
    print(f"Std:      {rent.std():.2f}")
    print(f"Min:      {rent.min()}")
    print(f"1%:       {rent.quantile(0.01)}")
    print(f"5%:       {rent.quantile(0.05)}")
    print(f"25% (Q1): {rent.quantile(0.25)}")
    print(f"50% (Med):{rent.median()}")
    print(f"75% (Q3): {rent.quantile(0.75)}")
    print(f"95%:      {rent.quantile(0.95)}")
    print(f"99%:      {rent.quantile(0.99)}")
    print(f"Max:      {rent.max()}")
    print(f"Skewness: {rent.skew():.4f}")
    print(f"Kurtosis: {rent.kurt():.4f}")

    # Check rent by FLAT_TYPE
    print("\nMedian and Mean Rent by Normalized FLAT_TYPE:")
    norm_ft = train["FLAT_TYPE"].str.strip().str.lower().str.replace(" ", "-")
    print(train.groupby(norm_ft)["MONTHLY_RENT"].agg(["count", "mean", "median", "min", "max"]))

if __name__ == "__main__":
    audit_dataset()

