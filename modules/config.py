"""
Paths and constants for the final pipeline (expt-2 V2 recipe with the honest test-quarter anchor).
Everything is resolved relative to the repository root, so the pipeline runs from any working directory.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
AUX_DIR = os.path.join(DATA_DIR, "auxiliary")

HDB_BLOCK_PATH = os.path.join(AUX_DIR, "sg-hdb-block.csv")
HDB_MEDIAN_RENT_PATH = os.path.join(AUX_DIR, "sg-hdb-median-rent.csv")
MRT_STATIONS_PATH = os.path.join(AUX_DIR, "sg-mrt-stations.csv")
SHOPPING_MALLS_PATH = os.path.join(AUX_DIR, "sg-shopping-malls.csv")
SCHOOLS_PATH = os.path.join(AUX_DIR, "sg-schools.csv")
COE_PRICES_PATH = os.path.join(AUX_DIR, "sg-coe-prices.csv")
STOCK_PRICES_PATH = os.path.join(AUX_DIR, "sg-stock-prices.csv")

RANDOM_SEED = 42
N_FOLDS = 5

FLAT_TYPE_ORDER = {
    "1-room": 1,
    "2-room": 2,
    "3-room": 3,
    "4-room": 4,
    "5-room": 5,
    "executive": 6,
}

MODEL_SYNONYMS = {
    "improved maisonette": "maisonette",
    "2 room": "2-room",
}

# Last quarter with labelled rents; test benchmarks after it are frozen and drifted (anchor.py).
CUTOFF_QUARTER = "2025-Q1"
# Out-of-time split used only to fit the blend weights (train <= 2024-06, validate >= 2024-07).
OOT_CUTOFF_DATE = "2024-07"
