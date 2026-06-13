"""
preprocessor.py
---------------
Build a sklearn ColumnTransformer pipeline for:
    - Numeric features  : median imputation → StandardScaler
    - Categorical features : most-frequent imputation → OrdinalEncoder

Returns a fitted preprocessor and the ordered feature names after transform,
which are required for correct SHAP index mapping.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build and return an *unfitted* ColumnTransformer.

    Numeric pipeline
    ----------------
    1. SimpleImputer(strategy='median')  – handles NaN
    2. StandardScaler()                  – zero-mean / unit-variance

    Categorical pipeline
    --------------------
    1. SimpleImputer(strategy='most_frequent')
    2. OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
       Tree models handle ordinal ints well; -1 signals "unseen" category.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,   NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_after_transform(preprocessor: ColumnTransformer) -> list[str]:
    """
    Return the column names in the same order as the transformed matrix.
    Must be called *after* fitting the preprocessor.
    """
    num_names = NUMERIC_FEATURES
    cat_names = CATEGORICAL_FEATURES
    return num_names + cat_names


def get_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    """
    Return a mapping  group_name → list of column indices in X_transformed.

    Parameters
    ----------
    feature_names : list[str]
        Ordered list returned by get_feature_names_after_transform().

    Returns
    -------
    dict with keys 'intrinsic' and 'extrinsic'.
    """
    from config import INTRINSIC_FEATURES, EXTRINSIC_FEATURES

    intrinsic_idx = [feature_names.index(f) for f in INTRINSIC_FEATURES
                     if f in feature_names]
    extrinsic_idx = [feature_names.index(f) for f in EXTRINSIC_FEATURES
                     if f in feature_names]

    return {
        "intrinsic": intrinsic_idx,
        "extrinsic": extrinsic_idx,
    }


def load_and_prepare(data_path: str, log_transform: bool = True):
    """
    Load CSV, select relevant columns, split X / y.

    Parameters
    ----------
    data_path     : str  – path to final_features_clean.csv
    log_transform : bool – if True, apply np.log1p to the target

    Returns
    -------
    X : pd.DataFrame   – feature matrix
    y : np.ndarray     – target vector (raw or log-transformed)
    y_raw : np.ndarray – always the raw price (for inverse-transform later)
    """
    from config import TARGET

    df = pd.read_csv(data_path)

    # Keep only the features we use + target
    cols_needed = ALL_FEATURES + [TARGET]
    df = df[cols_needed].copy()

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET])

    # Clip obvious outliers (price <= 0)
    df = df[df[TARGET] > 0].reset_index(drop=True)

    X = df[ALL_FEATURES]
    y_raw = df[TARGET].values.astype(float)
    y = np.log1p(y_raw) if log_transform else y_raw

    print(f"[Preprocessor] Loaded {len(df):,} samples | "
          f"features={len(ALL_FEATURES)} | "
          f"log_transform={log_transform}")

    return X, y, y_raw
