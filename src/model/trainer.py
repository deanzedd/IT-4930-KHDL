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
    XGB_PARAMS, CATBOOST_PARAMS, LGB_PARAMS, RF_PARAMS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, LOG_TRANSFORM,
    CATEGORICAL_FEATURES, TUNING_TRIALS,
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

def build_xgb_pipeline(preprocessor, params=None):
    from xgboost import XGBRegressor
    model = XGBRegressor(**(params or XGB_PARAMS))
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_lgb_pipeline(preprocessor, params=None):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(**(params or LGB_PARAMS))
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_rf_pipeline(preprocessor, params=None):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(**(params or RF_PARAMS))
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_catboost_pipeline(preprocessor, params=None):
    """
    CatBoost natively handles categoricals, but inside a sklearn Pipeline
    the preprocessor already OrdinalEncodes them, so cat_features is empty here.
    """
    model = CatBoostRegressor(**(params or CATBOOST_PARAMS))
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


# ─────────────────────────────────────────────
# Tuned ensemble training entry point
# ─────────────────────────────────────────────

def train_with_tuning(
    data_path    : str  = DATA_PATH,
    log_transform: bool = LOG_TRANSFORM,
    n_trials     : int  = TUNING_TRIALS,
) -> dict:
    """
    Full pipeline with Optuna tuning + 3-model weighted ensemble.

    Steps
    -----
    1. Load data, train/val/test split  (70% train, 10% val, 20% test)
    2. Tune XGBoost, LightGBM, RandomForest (Bayesian, n_trials each)
    3. Refit each model on train set with best params
    4. Optimise ensemble weights on val set
    5. Evaluate ensemble on test set
    6. Save all pipelines + report

    Returns
    -------
    dict with best individual pipelines, ensemble model, metrics, etc.
    """
    from tuner import tune_model
    from ensemble import EnsembleModel

    # ── 1. Load & split ─────────────────────────────────────
    X, y, y_raw = load_and_prepare(data_path, log_transform=log_transform)

    # First split: 80% train+val | 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Second split: 87.5% train | 12.5% val  => ~70% / 10% of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=RANDOM_STATE
    )
    print(f"[Trainer] Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}\n")

    # ── 2. Tune each model ─────────────────────────────────
    model_configs = [
        ("XGBoost",       build_xgb_pipeline),
        ("LightGBM",      build_lgb_pipeline),
        ("RandomForest",  build_rf_pipeline),
    ]

    print("=" * 60)
    print(f"Optuna Tuning ({n_trials} trials/model) — log_transform={log_transform}")
    print("=" * 60)

    best_params_all = {}
    for name, _ in model_configs:
        best_params, _ = tune_model(
            name, X_train, y_train,
            log_transform=log_transform,
            n_trials=n_trials,
        )
        best_params_all[name] = best_params

    # ── 3. Refit each model on FULL train set ─────────────────
    print("\n" + "=" * 60)
    print("Refitting tuned models on train set …")
    print("=" * 60)

    fitted_pipelines = {}
    cv_results = {}

    builder_map = {
        "XGBoost"      : build_xgb_pipeline,
        "LightGBM"     : build_lgb_pipeline,
        "RandomForest" : build_rf_pipeline,
    }

    for name, builder in model_configs:
        params = best_params_all[name]
        pipe   = builder(build_preprocessor(), params)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Quick CV on train set for reporting
        print(f"\n[{name}] Running {CV_FOLDS}-fold CV on train set …")
        cv_pipe = builder(build_preprocessor(), params)
        cv_metrics = cross_validate_model(cv_pipe, X_train, y_train, log_transform, name)
        cv_results[name] = cv_metrics

    # ── 4. Optimise ensemble weights on val set ──────────────
    print("\n" + "=" * 60)
    print("Optimising ensemble weights on validation set …")
    print("=" * 60)

    ensemble = EnsembleModel(fitted_pipelines)
    ensemble.fit_weights(X_val, y_val, log_transform=log_transform)

    # ── 5. Evaluate on test set ───────────────────────────
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    # Individual model test metrics
    individual_test_metrics = {}
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, log_transform)
        individual_test_metrics[name] = m
        print(f"  [{name}] RMSE={m['RMSE']:.4f} | R²={m['R2']:.4f} | MAPE={m['MAPE_%']:.2f}%")

    # Ensemble test metrics
    y_pred_ens = ensemble.predict(X_test)
    ensemble_test_metrics = compute_metrics(y_test, y_pred_ens, log_transform)
    print(f"\n  [Ensemble] RMSE={ensemble_test_metrics['RMSE']:.4f} | "
          f"R²={ensemble_test_metrics['R2']:.4f} | "
          f"MAPE={ensemble_test_metrics['MAPE_%']:.2f}%")

    # ── 6. Save all artifacts ────────────────────────────
    for name, pipe in fitted_pipelines.items():
        path = os.path.join(MODEL_DIR, f"{name.lower()}_tuned_pipeline.pkl")
        joblib.dump(pipe, path)
        print(f"[Trainer] Saved → {path}")

    ensemble_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    joblib.dump(ensemble, ensemble_path)
    print(f"[Trainer] Saved → {ensemble_path}")

    # Save extended report
    _save_ensemble_report(
        cv_results, individual_test_metrics, ensemble_test_metrics,
        ensemble.get_weights_dict(), best_params_all, log_transform
    )

    # Extract feature names from one of the fitted pipelines
    fitted_preprocessor = list(fitted_pipelines.values())[0].named_steps["preprocessor"]
    feature_names = get_feature_names_after_transform(fitted_preprocessor)

    return {
        "pipelines"              : fitted_pipelines,
        "ensemble"               : ensemble,
        "feature_names"          : feature_names,
        "X_train"                : X_train,
        "X_val"                  : X_val,
        "X_test"                 : X_test,
        "y_train"                : y_train,
        "y_val"                  : y_val,
        "y_test"                 : y_test,
        "cv_results"             : cv_results,
        "individual_test_metrics": individual_test_metrics,
        "ensemble_test_metrics"  : ensemble_test_metrics,
        "best_params"            : best_params_all,
        "log_transform"          : log_transform,
    }


def _save_ensemble_report(
    cv_results: dict,
    individual_test_metrics: dict,
    ensemble_test_metrics: dict,
    weights: dict,
    best_params: dict,
    log_transform: bool,
):
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    sep = "=" * 60
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(sep + "\n")
        f.write("Tuned Ensemble Model — Evaluation Report\n")
        f.write(sep + "\n\n")
        f.write(f"Log-transform target : {log_transform}\n")
        f.write(f"Models               : {', '.join(cv_results.keys())}\n\n")

        f.write("─" * 60 + "\n")
        f.write("── Cross-Validation Results (tuned params, train set) ──\n")
        f.write("─" * 60 + "\n")
        for name, m in cv_results.items():
            f.write(f"\n  {name}:\n")
            for k, v in m.items():
                f.write(f"    {k}: {v:.4f}\n")

        f.write("\n" + "─" * 60 + "\n")
        f.write("── Individual Test Set Metrics ──\n")
        f.write("─" * 60 + "\n")
        for name, m in individual_test_metrics.items():
            f.write(f"\n  {name}:\n")
            for k, v in m.items():
                f.write(f"    {k}: {v:.4f}\n")

        f.write("\n" + "─" * 60 + "\n")
        f.write("── Ensemble Test Set Metrics ──\n")
        f.write("─" * 60 + "\n")
        for k, v in ensemble_test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")

        f.write("\n" + "─" * 60 + "\n")
        f.write("── Ensemble Weights ──\n")
        f.write("─" * 60 + "\n")
        for name, w in weights.items():
            f.write(f"  {name}: {w:.4f}\n")

        f.write("\n" + "─" * 60 + "\n")
        f.write("── Best Hyperparameters ──\n")
        f.write("─" * 60 + "\n")
        for name, params in best_params.items():
            f.write(f"\n  {name}:\n")
            for k, v in params.items():
                f.write(f"    {k}: {v}\n")

    print(f"[Trainer] Report saved → {report_path}")


if __name__ == "__main__":
    results = train()
