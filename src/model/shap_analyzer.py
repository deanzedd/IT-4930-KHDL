"""
shap_analyzer.py
----------------
SHAP-based explanation and price decomposition.

Key formula
-----------
    Price_pred  ≈  base_value
                 + Σ SHAP_intrinsic
                 + Σ SHAP_extrinsic

All analysis is done on the *transformed* feature matrix (output of the
fitted preprocessor), so SHAP indices must be mapped through feature_names.

Public API
----------
    SHAPAnalyzer(pipeline, feature_names, log_transform)
        .compute_shap_values(X)           → shap.Explanation
        .decompose(shap_vals, sample_idx) → dict
        .decompose_dataset(shap_vals, X)  → pd.DataFrame
        .plot_waterfall(shap_vals, X_raw, sample_idx, title)
        .plot_group_bar(shap_vals, X_raw)
        .plot_beeswarm(shap_vals, group)
        .plot_group_scatter(shap_vals, X_raw)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless / non-interactive backend
import matplotlib.pyplot as plt
import shap

from config import (
    INTRINSIC_FEATURES,
    EXTRINSIC_FEATURES,
    OUTPUT_DIR,
)
from preprocessor import get_group_indices


warnings.filterwarnings("ignore")


class SHAPAnalyzer:
    """
    Wrap a fitted sklearn Pipeline (preprocessor + tree model) and provide
    SHAP-based decomposition into Intrinsic and Extrinsic price components.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    feature_names : list[str] – ordered column names after preprocessing
                    (from preprocessor.get_feature_names_after_transform)
    log_transform : bool – whether the model was trained on log1p(price)
    """

    def __init__(self, pipeline, feature_names: list[str],
                 log_transform: bool = True):
        self.pipeline      = pipeline
        self.preprocessor  = pipeline.named_steps["preprocessor"]
        self.model         = pipeline.named_steps["model"]
        self.feature_names = feature_names
        self.log_transform = log_transform

        # Group indices in the transformed matrix
        self.group_idx = get_group_indices(feature_names)
        self.intrinsic_idx = self.group_idx["intrinsic"]
        self.extrinsic_idx = self.group_idx["extrinsic"]

        # SHAP explainer (built lazily)
        self._explainer = None

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _get_explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessor and return a dense numpy array."""
        X_t = self.preprocessor.transform(X)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        return np.array(X_t, dtype=float)

    # ─────────────────────────────────────────────────────────
    # SHAP computation
    # ─────────────────────────────────────────────────────────

    def compute_shap_values(self, X: pd.DataFrame) -> shap.Explanation:
        """
        Compute SHAP values for every sample in X.

        Returns a shap.Explanation object whose .values has shape
        (n_samples, n_features) and .base_values has shape (n_samples,).

        Note: values are in the model's *output* space (log-scale if
        log_transform=True). Use decompose() for human-readable results.
        """
        X_t     = self._transform(X)
        exp     = self._get_explainer()
        shap_ex = exp(X_t)

        # Attach feature names for nicer plots
        shap_ex.feature_names = self.feature_names
        return shap_ex

    # ─────────────────────────────────────────────────────────
    # Decomposition
    # ─────────────────────────────────────────────────────────

    def decompose(self, shap_explanation: shap.Explanation,
                  sample_idx: int) -> dict:
        """
        Break down ONE sample's predicted price into components.

        Returns
        -------
        dict with keys:
            base_value         – model baseline (log-scale or price-scale)
            intrinsic_contrib  – sum of SHAP for intrinsic features
            extrinsic_contrib  – sum of SHAP for extrinsic features
            predicted_log      – raw model output (log-scale if applicable)
            predicted_price    – predicted price in tỷ đồng
            check_sum          – should ≈ predicted_log (sanity check)
        """
        sv   = shap_explanation.values[sample_idx]           # (n_features,)
        base = float(shap_explanation.base_values[sample_idx])

        intr  = float(sv[self.intrinsic_idx].sum())
        extr  = float(sv[self.extrinsic_idx].sum())
        total = base + intr + extr                            # ≈ predicted_log

        if self.log_transform:
            pred_price = float(np.expm1(total))
            base_price = float(np.expm1(base))
        else:
            pred_price = total
            base_price = base

        return {
            "base_value"        : base,
            "base_price_bilion" : base_price,
            "intrinsic_contrib" : intr,
            "extrinsic_contrib" : extr,
            "predicted_log"     : total,
            "predicted_price"   : pred_price,
            "check_sum"         : total,          # base + intr + extr
        }

    def decompose_dataset(self, shap_explanation: shap.Explanation,
                          X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute decomposition for *every* sample and return a DataFrame.

        Columns: base_value, intrinsic_contrib, extrinsic_contrib,
                 predicted_log, predicted_price
        """
        sv   = shap_explanation.values                       # (N, F)
        base = shap_explanation.base_values                  # (N,)

        intr = sv[:, self.intrinsic_idx].sum(axis=1)
        extr = sv[:, self.extrinsic_idx].sum(axis=1)
        total = base + intr + extr

        df = pd.DataFrame({
            "base_value"        : base,
            "intrinsic_contrib" : intr,
            "extrinsic_contrib" : extr,
            "predicted_log"     : total,
            "predicted_price"   : np.expm1(total) if self.log_transform else total,
        })
        return df

    # ─────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────

    def plot_waterfall(self, shap_explanation: shap.Explanation,
                       X_raw: pd.DataFrame,
                       sample_idx: int = 0,
                       title: str = "SHAP Waterfall – Sample",
                       save_name: str = "shap_waterfall.png"):
        """
        Standard SHAP waterfall chart for a single property.
        Features are shown in the model's output (log) space.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap_explanation[sample_idx],
            max_display=20,
            show=False,
        )
        plt.title(title, fontsize=13)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Waterfall saved → {path}")

    def plot_group_bar(self, shap_explanation: shap.Explanation,
                       X_raw: pd.DataFrame,
                       save_name: str = "shap_group_bar.png"):
        """
        Horizontal bar chart: mean |SHAP| contribution per group (Intrinsic / Extrinsic).
        """
        sv = shap_explanation.values                        # (N, F)

        intr_mean = np.abs(sv[:, self.intrinsic_idx]).mean(axis=0)
        extr_mean = np.abs(sv[:, self.extrinsic_idx]).mean(axis=0)

        intr_names = [self.feature_names[i] for i in self.intrinsic_idx]
        extr_names = [self.feature_names[i] for i in self.extrinsic_idx]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Mean |SHAP| by Feature Group", fontsize=14, fontweight="bold")

        # ─ Intrinsic ─
        order_i = np.argsort(intr_mean)
        axes[0].barh([intr_names[j] for j in order_i],
                     intr_mean[order_i], color="#4C72B0")
        axes[0].set_title("Intrinsic (Physical)", fontsize=12)
        axes[0].set_xlabel("Mean |SHAP value|")

        # ─ Extrinsic ─
        order_e = np.argsort(extr_mean)
        axes[1].barh([extr_names[j] for j in order_e],
                     extr_mean[order_e], color="#DD8452")
        axes[1].set_title("Extrinsic (Location & Amenity)", fontsize=12)
        axes[1].set_xlabel("Mean |SHAP value|")

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Group bar chart saved → {path}")

    def plot_beeswarm(self, shap_explanation: shap.Explanation,
                      group: str = "intrinsic",
                      save_name: str = None):
        """
        Beeswarm summary plot for ONE group (intrinsic or extrinsic).

        Parameters
        ----------
        group : 'intrinsic' or 'extrinsic'
        """
        idx  = self.intrinsic_idx if group == "intrinsic" else self.extrinsic_idx
        names = [self.feature_names[i] for i in idx]

        # Slice the explanation object to the group features
        sv_group   = shap_explanation.values[:, idx]
        data_group = shap_explanation.data[:, idx] if shap_explanation.data is not None else None
        base_vals  = shap_explanation.base_values

        sub_exp = shap.Explanation(
            values       = sv_group,
            base_values  = base_vals,
            data         = data_group,
            feature_names= names,
        )

        save_name = save_name or f"shap_beeswarm_{group}.png"

        fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.5 + 2)))
        shap.plots.beeswarm(sub_exp, max_display=len(names), show=False)
        plt.title(f"SHAP Beeswarm – {group.capitalize()} Features", fontsize=13)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Beeswarm ({group}) saved → {path}")

    def plot_group_scatter(self, shap_explanation: shap.Explanation,
                           X_raw: pd.DataFrame,
                           save_name: str = "shap_group_scatter.png"):
        """
        Scatter plot: Σ SHAP_intrinsic vs Σ SHAP_extrinsic per sample.
        Color = predicted price.
        """
        sv   = shap_explanation.values
        intr = sv[:, self.intrinsic_idx].sum(axis=1)
        extr = sv[:, self.extrinsic_idx].sum(axis=1)
        base = shap_explanation.base_values
        pred = np.expm1(base + intr + extr) if self.log_transform else (base + intr + extr)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(intr, extr, c=pred, cmap="viridis", alpha=0.5, s=15)
        plt.colorbar(sc, ax=ax, label="Predicted Price (tỷ đồng)")
        ax.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax.axvline(0, color="gray", lw=0.8, linestyle="--")
        ax.set_xlabel("Σ SHAP – Intrinsic")
        ax.set_ylabel("Σ SHAP – Extrinsic")
        ax.set_title("Intrinsic vs Extrinsic SHAP Contributions", fontsize=13)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Group scatter saved → {path}")

    def plot_group_importance_summary(self,
                                      shap_explanation: shap.Explanation,
                                      save_name: str = "shap_group_importance_summary.png"):
        """
        Two-bar summary: total mean |SHAP| for Intrinsic group vs Extrinsic group.
        Shows which group drives price prediction more overall.
        """
        sv = shap_explanation.values
        intr_total = np.abs(sv[:, self.intrinsic_idx]).sum(axis=1).mean()
        extr_total = np.abs(sv[:, self.extrinsic_idx]).sum(axis=1).mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Intrinsic\n(Physical)", "Extrinsic\n(Location & Amenity)"],
                      [intr_total, extr_total],
                      color=["#4C72B0", "#DD8452"], width=0.5)
        for bar, val in zip(bars, [intr_total, extr_total]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("Mean |SHAP| (sum over group features)")
        ax.set_title("Group-level SHAP Importance", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Group importance summary saved → {path}")

    def print_sample_decomposition(self, shap_explanation: shap.Explanation,
                                   X_raw: pd.DataFrame,
                                   sample_idx: int = 0):
        """Pretty-print price decomposition for one sample."""
        d = self.decompose(shap_explanation, sample_idx)
        row = X_raw.iloc[sample_idx]

        print("\n" + "=" * 55)
        print(f"  Price Decomposition — Sample #{sample_idx}")
        print("=" * 55)
        print(f"  District      : {row.get('district', 'N/A')}")
        print(f"  Property type : {row.get('property_type', 'N/A')}")
        print(f"  Area          : {row.get('area', 'N/A')} m²")
        print("-" * 55)
        print(f"  Baseline price          : {d['base_price_bilion']:>8.3f} tỷ")
        print(f"  + Intrinsic contribution: {d['intrinsic_contrib']:>+8.4f}  (log-scale)")
        print(f"  + Extrinsic contribution: {d['extrinsic_contrib']:>+8.4f}  (log-scale)")
        print("-" * 55)
        print(f"  Predicted price         : {d['predicted_price']:>8.3f} tỷ đồng")
        print(f"  Sanity check (sum)      : {d['check_sum']:>8.6f} "
              f"≈ predicted_log {d['predicted_log']:.6f}")
        print("=" * 55 + "\n")
