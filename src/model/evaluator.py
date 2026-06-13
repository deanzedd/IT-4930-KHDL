"""
evaluator.py
------------
Post-training evaluation utilities:
    - Metrics on test set (RMSE / MAE / R² / MAPE)
    - Predicted vs Actual scatter plot
    - Residual distribution plot
    - Native feature importance plot (XGBoost gain / CatBoost)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import OUTPUT_DIR


def evaluate_and_plot(pipeline,
                      X_test: pd.DataFrame,
                      y_test: np.ndarray,
                      log_transform: bool = True,
                      model_name: str = "Model") -> dict:
    """
    Full evaluation on the held-out test set.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    X_test        : test features (raw, pre-preprocessor)
    y_test        : test targets (log-scale if log_transform=True)
    log_transform : if True, apply np.expm1 before computing metrics
    model_name    : used for plot titles and file names

    Returns
    -------
    dict with RMSE, MAE, R2, MAPE_% on the original (tỷ đồng) scale.
    """
    y_pred = pipeline.predict(X_test)

    # Convert to original scale for human-readable metrics
    if log_transform:
        y_true_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_test
        y_pred_orig = y_pred

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    mae  = float(mean_absolute_error(y_true_orig, y_pred_orig))
    r2   = float(r2_score(y_true_orig, y_pred_orig))
    mask = y_true_orig != 0
    mape = float(np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask])
                                 / y_true_orig[mask])) * 100)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_%": mape}

    print(f"\n[Evaluator] ── {model_name} Test Metrics ──")
    print(f"  RMSE  : {rmse:.4f} tỷ")
    print(f"  MAE   : {mae:.4f} tỷ")
    print(f"  R²    : {r2:.4f}")
    print(f"  MAPE  : {mape:.2f}%")

    _plot_predicted_vs_actual(y_true_orig, y_pred_orig, metrics, model_name)
    _plot_residuals(y_true_orig, y_pred_orig, model_name)
    _plot_feature_importance(pipeline, model_name)

    return metrics


# ─────────────────────────────────────────────────────────────────
# Internal plot helpers
# ─────────────────────────────────────────────────────────────────

def _plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                               metrics: dict, model_name: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color="#4C72B0", label="Samples")

    # Perfect-prediction line
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")

    ax.set_xlabel("Actual Price (tỷ đồng)", fontsize=11)
    ax.set_ylabel("Predicted Price (tỷ đồng)", fontsize=11)
    ax.set_title(f"{model_name} — Predicted vs Actual\n"
                 f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.3f}  "
                 f"MAPE={metrics['MAPE_%']:.2f}%",
                 fontsize=11)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"prediction_vs_actual_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Plot saved → {path}")


def _plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{model_name} — Residual Analysis", fontsize=13, fontweight="bold")

    # ── Residuals vs Predicted ──
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="#DD8452")
    axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Price (tỷ đồng)")
    axes[0].set_ylabel("Residual (tỷ đồng)")
    axes[0].set_title("Residuals vs Predicted")

    # ── Residual histogram ──
    axes[1].hist(residuals, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (tỷ đồng)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"residuals_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Plot saved → {path}")


def _plot_feature_importance(pipeline, model_name: str):
    """
    Plot native feature importance (XGBoost gain or CatBoost feature_importances_).
    Silently skips if the model type is not recognised.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    from preprocessor import get_feature_names_after_transform
    feature_names = get_feature_names_after_transform(preprocessor)

    importances = None
    importance_type = "Importance"

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # XGBoost gain (more informative than weight)
        try:
            gain = model.get_booster().get_score(importance_type="gain")
            if gain:
                importances = np.array([gain.get(f"f{i}", 0)
                                        for i in range(len(feature_names))])
                importance_type = "Gain"
        except Exception:
            pass

    if importances is None:
        return

    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names) * 0.45 + 2)))
    ax.barh([feature_names[i] for i in order],
            importances[order], color="#4C72B0")
    ax.set_xlabel(f"Feature {importance_type}")
    ax.set_title(f"{model_name} — Feature Importance ({importance_type})", fontsize=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"feature_importance_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Plot saved → {path}")
