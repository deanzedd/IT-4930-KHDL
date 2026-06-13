"""
config.py
---------
Central configuration for the Unified Tree-based Model pipeline.

Feature groups:
    - INTRINSIC : Physical attributes of the property
    - EXTRINSIC : Location & amenity features
    Target: price_billion (optionally log-transformed)
"""

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH  = os.path.join(ROOT_DIR, "data", "final_features_clean.csv")
MODEL_DIR  = os.path.join(ROOT_DIR, "src", "model", "saved_models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "src", "model", "outputs")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Target
# ─────────────────────────────────────────────
TARGET = "price_billion"

# ─────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────

# Nhóm 1 – Intrinsic (Đặc điểm vật lý)
INTRINSIC_FEATURES = [
    "area",
    "property_type",
    "bedrooms_detail",
    "bathrooms_detail",
    "floors",
    "width_m",
    "depth_m",
]

# Nhóm 2 – Extrinsic (Vị trí & Tiện ích)
EXTRINSIC_FEATURES = [
    "district",
    "is_pho_co",
    "dist_hoan_kiem_km",
    "dist_hospital_nearest_km",
    "hospital_count_2km",
    "dist_university_nearest_km",
    "university_count_2km",
    "dist_mall_nearest_km",
    "mall_count_2km",
    "dist_lake_nearest_km",
    "lake_count_2km",
]

ALL_FEATURES = INTRINSIC_FEATURES + EXTRINSIC_FEATURES

# Categorical features (for encoding)
CATEGORICAL_FEATURES = ["property_type", "district"]

# Numeric features (for scaling)
NUMERIC_FEATURES = [f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES]

# ─────────────────────────────────────────────
# Training settings
# ─────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.20          # 80 / 20 split
CV_FOLDS        = 5
LOG_TRANSFORM   = True          # Try log1p(price_billion) as target

# ─────────────────────────────────────────────
# XGBoost hyperparameters (baseline)
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators"     : 500,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_weight" : 3,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
    "verbosity"        : 0,
}

# ─────────────────────────────────────────────
# CatBoost hyperparameters (baseline)
# ─────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations"     : 500,
    "depth"          : 6,
    "learning_rate"  : 0.05,
    "l2_leaf_reg"    : 3,
    "random_seed"    : RANDOM_STATE,
    "verbose"        : 0,
    "allow_writing_files": False,
}
