"""
ensemble.py
-----------
Weighted-average ensemble of 3 fitted sklearn Pipelines.

Weights are automatically optimised on a held-out validation split
using scipy's SLSQP minimiser (minimise RMSE subject to w ≥ 0, sum(w)=1).

Usage
-----
    from ensemble import EnsembleModel
    ens = EnsembleModel({"XGBoost": pipe_xgb, "LightGBM": pipe_lgb, "RandomForest": pipe_rf})
    ens.fit_weights(X_val, y_val, log_transform=True)
    y_pred = ens.predict(X_test)
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from trainer import compute_metrics


class EnsembleModel:
    """
    Weighted-average ensemble of multiple fitted sklearn pipelines.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        Mapping model_name -> fitted sklearn Pipeline.
    """

    def __init__(self, pipelines: dict):
        if len(pipelines) < 2:
            raise ValueError("Need at least 2 pipelines for an ensemble.")
        self.pipelines  = pipelines
        self.names      = list(pipelines.keys())
        n               = len(self.names)
        self.weights    = np.full(n, 1.0 / n)   # equal weights initially
        self._fitted    = False

    # ── Prediction ──────────────────────────────────────────────

    def _predict_all(self, X) -> np.ndarray:
        """Return (n_models, n_samples) matrix of raw predictions."""
        preds = []
        for name in self.names:
            preds.append(self.pipelines[name].predict(X))
        return np.vstack(preds)  # shape: (n_models, n_samples)

    def predict(self, X) -> np.ndarray:
        """Weighted average prediction."""
        pred_matrix = self._predict_all(X)
        return self.weights @ pred_matrix  # (n_models,) @ (n_models, n_samples)

    # ── Weight optimisation ─────────────────────────────────────

    def fit_weights(self, X_val, y_val, log_transform: bool = True):
        """
        Optimise ensemble weights on (X_val, y_val) to minimise RMSE.

        Uses SLSQP with constraints: weights >= 0, sum(weights) == 1.
        """
        pred_matrix = self._predict_all(X_val)   # (n_models, n_samples)
        n_models = len(self.names)

        def rmse_objective(w):
            y_pred = w @ pred_matrix
            metrics = compute_metrics(y_val, y_pred, log_transform)
            return metrics["RMSE"]

        # Constraints: sum(w) == 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        # Bounds: 0 <= w_i <= 1
        bounds = [(0.0, 1.0)] * n_models
        # Start from equal weights
        w0 = np.full(n_models, 1.0 / n_models)

        result = minimize(
            rmse_objective, w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success:
            self.weights = result.x
        else:
            print(f"[Ensemble] ⚠️  Weight optimisation did not converge: {result.message}")
            print("[Ensemble]    Falling back to equal weights.")

        self._fitted = True
        print(f"\n[Ensemble] Optimised weights:")
        for name, w in zip(self.names, self.weights):
            print(f"  {name:15s}: {w:.4f}")

        # Report ensemble CV metrics on validation set
        y_pred_val = self.predict(X_val)
        m = compute_metrics(y_val, y_pred_val, log_transform)
        print(f"[Ensemble] Validation RMSE: {m['RMSE']:.4f} | "
              f"R²: {m['R2']:.4f} | MAPE: {m['MAPE_%']:.2f}%")
        return self

    # ── sklearn-compatible interface ─────────────────────────────

    def get_weights_dict(self) -> dict:
        return {name: float(w) for name, w in zip(self.names, self.weights)}
