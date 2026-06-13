"""
trainer.py
----------
Train XGBoost and CatBoost regressors inside a sklearn Pipeline.

Workflow
--------
1. Build preprocessor (ColumnTransformer)
2. Wrap in sklearn Pipeline + model
3. 5-fold CV to evaluate both models
4. Compare → select winner
5. Refit winner on full train set
6. Serialize model + preprocessor to disk

Supports both raw price and log1p(price) targets.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[Trainer] WARNING: xgboost not installed. Skipping XGBoost.")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("[Trainer] WARNING: catboost not installed. Skipping CatBoost.")

from config import (
    DATA_PATH, MODEL_DIR, OUTPUT_DIR,
    XGB_PARAMS, CATBOOST_PARAMS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, LOG_TRANSFORM,
    CATEGORICAL_FEATURES,
)
from preprocessor import build_preprocessor, load_and_prepare, get_feature_names_after_transform


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    log_transform: bool = False) -> dict:
    """
    Compute RMSE / MAE / R² / MAPE on *original* scale.
    If log_transform=True, inverse-transform predictions first.
    """
    if log_transform:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape_val = mape(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_%": mape_val}


# ─────────────────────────────────────────────────────────────
# Build model pipelines
# ─────────────────────────────────────────────────────────────

def build_xgb_pipeline(preprocessor):
    model = XGBRegressor(**XGB_PARAMS)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_catboost_pipeline(preprocessor):
    """
    CatBoost natively handles categoricals, but inside a sklearn Pipeline
    the preprocessor already OrdinalEncodes them, so cat_features is empty here.
    """
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ─────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────

def cross_validate_model(pipeline, X: pd.DataFrame, y: np.ndarray,
                          log_transform: bool, label: str) -> dict:
    """
    Run KFold CV and return averaged metrics on original price scale.
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_val)

        metrics = compute_metrics(y_val, y_pred, log_transform)
        fold_metrics.append(metrics)
        print(f"  [{label}] Fold {fold}/{CV_FOLDS} | "
              f"RMSE={metrics['RMSE']:.3f} | "
              f"MAE={metrics['MAE']:.3f} | "
              f"R²={metrics['R2']:.4f} | "
              f"MAPE={metrics['MAPE_%']:.2f}%")

    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print(f"  [{label}] ── CV Average ── "
          f"RMSE={avg['RMSE']:.3f} | MAE={avg['MAE']:.3f} | "
          f"R²={avg['R2']:.4f} | MAPE={avg['MAPE_%']:.2f}%\n")
    return avg


# ─────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────

def train(data_path: str = DATA_PATH,
          log_transform: bool = LOG_TRANSFORM) -> dict:
    """
    Full training workflow.

    Returns
    -------
    results : dict with keys
        'best_model_name', 'best_pipeline', 'preprocessor',
        'feature_names', 'X_test', 'y_test',
        'cv_results', 'test_metrics', 'log_transform'
    """
    # ── 1. Load data ──────────────────────────────────────────
    X, y, y_raw = load_and_prepare(data_path, log_transform=log_transform)

    # ── 2. Train / test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Trainer] Train={len(X_train):,}  Test={len(X_test):,}\n")

    # ── 3. Build pipelines ────────────────────────────────────
    candidates = {}
    if XGB_AVAILABLE:
        candidates["XGBoost"]  = build_xgb_pipeline(build_preprocessor())
    if CATBOOST_AVAILABLE:
        candidates["CatBoost"] = build_catboost_pipeline(build_preprocessor())

    if not candidates:
        raise RuntimeError("Neither XGBoost nor CatBoost is installed!")

    # ── 4. Cross-validation on train set ─────────────────────
    cv_results = {}
    print("=" * 60)
    print(f"Cross-Validation ({CV_FOLDS}-Fold) — log_transform={log_transform}")
    print("=" * 60)
    for name, pipe in candidates.items():
        print(f"\n[{name}]")
        cv_results[name] = cross_validate_model(pipe, X_train, y_train,
                                                  log_transform, name)

    # ── 5. Pick winner (lowest RMSE) ─────────────────────────
    best_name = min(cv_results, key=lambda k: cv_results[k]["RMSE"])
    print(f"\n[Trainer] ✅ Best model by RMSE: {best_name}")

    # ── 6. Refit winner on FULL train set ─────────────────────
    print(f"[Trainer] Refitting {best_name} on full train set …")
    best_pipeline = candidates[best_name]

    # Rebuild a fresh pipeline (CV already called fit multiple times)
    if best_name == "XGBoost":
        best_pipeline = build_xgb_pipeline(build_preprocessor())
    else:
        best_pipeline = build_catboost_pipeline(build_preprocessor())

    best_pipeline.fit(X_train, y_train)

    # ── 7. Test-set evaluation ────────────────────────────────
    y_pred_test = best_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test, log_transform)
    print("\n[Trainer] ── Test Set Metrics ──")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── 8. Save pipeline & preprocessor separately ─────────────
    pipeline_path     = os.path.join(MODEL_DIR, f"{best_name.lower()}_pipeline.pkl")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

    joblib.dump(best_pipeline, pipeline_path)
    joblib.dump(best_pipeline.named_steps["preprocessor"], preprocessor_path)
    print(f"\n[Trainer] Saved → {pipeline_path}")
    print(f"[Trainer] Saved → {preprocessor_path}")

    # ── 9. Save all CV results to text ────────────────────────
    _save_cv_report(cv_results, test_metrics, best_name, log_transform)

    # Extract fitted preprocessor for SHAP use
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names_after_transform(fitted_preprocessor)

    return {
        "best_model_name" : best_name,
        "best_pipeline"   : best_pipeline,
        "preprocessor"    : fitted_preprocessor,
        "feature_names"   : feature_names,
        "X_train"         : X_train,
        "X_test"          : X_test,
        "y_train"         : y_train,
        "y_test"          : y_test,
        "cv_results"      : cv_results,
        "test_metrics"    : test_metrics,
        "log_transform"   : log_transform,
    }


def _save_cv_report(cv_results: dict, test_metrics: dict,
                    best_name: str, log_transform: bool):
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Unified Tree-based Model — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Log-transform target: {log_transform}\n")
        f.write(f"Best model: {best_name}\n\n")

        f.write("── Cross-Validation Results ──\n")
        for name, metrics in cv_results.items():
            f.write(f"\n  {name}:\n")
            for k, v in metrics.items():
                f.write(f"    {k}: {v:.4f}\n")

        f.write("\n── Test Set Metrics ──\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")

    print(f"[Trainer] Report saved → {report_path}")


if __name__ == "__main__":
    results = train()
