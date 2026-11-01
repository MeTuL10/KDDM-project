"""
Configuration and constants for Singapore HDB Monthly Rent Prediction (Experiment 2).
"""

import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# expt-2 now lives in exploration/Aayush/, so the repository root is three levels up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
AUX_DIR = os.path.join(DATA_DIR, "auxiliary")

# Auxiliary File Paths
HDB_BLOCK_PATH = os.path.join(AUX_DIR, "sg-hdb-block.csv")
HDB_MEDIAN_RENT_PATH = os.path.join(AUX_DIR, "sg-hdb-median-rent.csv")
MRT_STATIONS_PATH = os.path.join(AUX_DIR, "sg-mrt-stations.csv")
SHOPPING_MALLS_PATH = os.path.join(AUX_DIR, "sg-shopping-malls.csv")
SCHOOLS_PATH = os.path.join(AUX_DIR, "sg-schools.csv")
COE_PRICES_PATH = os.path.join(AUX_DIR, "sg-coe-prices.csv")
STOCK_PRICES_PATH = os.path.join(AUX_DIR, "sg-stock-prices.csv")

# Output Directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CLEANED_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned")
FEATURE_DATA_DIR = os.path.join(OUTPUT_DIR, "features")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
SUBMISSION_DIR = os.path.join(OUTPUT_DIR, "submissions")

# Singapore Downtown Core / CBD Reference Coordinates (Raffles Place)
CBD_LAT = 1.2838
CBD_LON = 103.8516

# Earth radius in kilometers for Haversine calculations
EARTH_RADIUS_KM = 6371.0088

# Random Seed for Reproducibility
RANDOM_STATE = 42
RANDOM_SEED = 42
N_FOLDS = 5

# Categorical Standardization Mappings
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

# Validation Split Settings
OOT_CUTOFF_DATE = "2024-07"  # Train <= 2024-06, Val >= 2024-07
N_SPLITS_KFOLD = 5

