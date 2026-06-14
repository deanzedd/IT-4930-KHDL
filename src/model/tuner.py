"""
tuner.py
--------
Optuna-based hyperparameter tuner for XGBoost, LightGBM, and RandomForest.

Usage
-----
    from tuner import tune_model
    best_params = tune_model("XGBoost", X_train, y_train, log_transform=True)
"""

import warnings
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from config import RANDOM_STATE, TUNING_TRIALS, TUNING_CV_FOLDS
from preprocessor import build_preprocessor
from trainer import compute_metrics


# ─────────────────────────────────────────────────────────────
# Search spaces
# ─────────────────────────────────────────────────────────────

def _xgb_space(trial) -> dict:
    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth"        : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
        "verbosity"        : 0,
    }


def _lgb_space(trial) -> dict:
    return {
        "n_estimators"      : trial.suggest_int("n_estimators", 200, 1000, step=50),
        "max_depth"         : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves"        : trial.suggest_int("num_leaves", 20, 150),
        "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples" : trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state"      : RANDOM_STATE,
        "n_jobs"            : -1,
        "verbosity"         : -1,
    }


def _rf_space(trial) -> dict:
    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth"        : trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features"     : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
    }


# ─────────────────────────────────────────────────────────────
# Objective factory
# ─────────────────────────────────────────────────────────────

def _make_objective(model_name: str, X, y, log_transform: bool):
    """
    Returns an Optuna objective function that runs KFold CV
    and returns mean RMSE (on original price scale).
    """
    def objective(trial):
        if model_name == "XGBoost":
            from xgboost import XGBRegressor
            params = _xgb_space(trial)
            model  = XGBRegressor(**params)

        elif model_name == "LightGBM":
            from lightgbm import LGBMRegressor
            params = _lgb_space(trial)
            model  = LGBMRegressor(**params)

        elif model_name == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            params = _rf_space(trial)
            model  = RandomForestRegressor(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ])

        kf = KFold(n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        rmse_scores = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)
            metrics = compute_metrics(y_val, y_pred, log_transform)
            rmse_scores.append(metrics["RMSE"])

        return float(np.mean(rmse_scores))

    return objective


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def tune_model(
    model_name   : str,
    X_train,
    y_train,
    log_transform: bool = True,
    n_trials     : int  = TUNING_TRIALS,
    verbose      : bool = True,
) -> dict:
    """
    Run Optuna hyperparameter search for the given model.

    Parameters
    ----------
    model_name   : "XGBoost" | "LightGBM" | "RandomForest"
    X_train      : pd.DataFrame — training features
    y_train      : np.ndarray  — training target (possibly log-transformed)
    log_transform: bool        — whether target is log1p(price)
    n_trials     : int         — number of Bayesian optimization trials
    verbose      : bool        — print progress

    Returns
    -------
    best_params : dict — best hyperparameters found
    """
    if verbose:
        print(f"\n[Tuner] ⚙️  Tuning {model_name} ({n_trials} trials, {TUNING_CV_FOLDS}-fold CV) …")

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    objective = _make_objective(model_name, X_train, y_train, log_transform)

    study.optimize(
        objective,
        n_trials  = n_trials,
        n_jobs    = 1,          # sequential to avoid data race
        show_progress_bar = verbose,
    )

    best_params = study.best_params
    best_rmse   = study.best_value

    if verbose:
        print(f"[Tuner] ✅ {model_name} best CV-RMSE: {best_rmse:.4f}")
        print(f"[Tuner]    Best params: {best_params}\n")

    # Add fixed params back
    if model_name == "XGBoost":
        best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
    elif model_name == "LightGBM":
        best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    elif model_name == "RandomForest":
        best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1})

    return best_params, study
