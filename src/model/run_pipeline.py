"""
run_pipeline.py
---------------
Entry point for the complete modeling pipeline.

Usage
-----
    # From the src/model/ directory:
    python run_pipeline.py

    # Compare log vs raw target:
    python run_pipeline.py --compare-log-transform

Steps
-----
    1. Load & validate data
    2. (Optional) Compare log1p vs raw target via CV
    3. Train XGBoost + CatBoost → select winner
    4. Evaluate on test set
    5. Compute SHAP values → Intrinsic / Extrinsic decomposition
    6. Generate all plots
    7. Print sample decompositions
    8. Save all artifacts
"""

import sys
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Make sure imports resolve when running from src/model/ ──
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_PATH, MODEL_DIR, OUTPUT_DIR, LOG_TRANSFORM
from preprocessor import load_and_prepare, build_preprocessor, get_feature_names_after_transform
from trainer import train
from shap_analyzer import SHAPAnalyzer
from evaluator import evaluate_and_plot


def compare_log_transforms(data_path: str = DATA_PATH):
    """
    Quick 5-fold CV comparison: raw price vs log1p(price) target.
    Prints R² and MAPE for both settings.
    """
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor
    from config import XGB_PARAMS, RANDOM_STATE, CV_FOLDS
    from trainer import compute_metrics

    print("\n" + "=" * 60)
    print("Comparing log-transform settings on XGBoost (5-fold CV)")
    print("=" * 60)

    for use_log in [True, False]:
        X, y, _ = load_and_prepare(data_path, log_transform=use_log)
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_r2, fold_mape = [], []

        for tr, val in kf.split(X):
            pipe = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("model", XGBRegressor(**XGB_PARAMS)),
            ])
            pipe.fit(X.iloc[tr], y[tr])
            y_pred = pipe.predict(X.iloc[val])
            m = compute_metrics(y[val], y_pred, use_log)
            fold_r2.append(m["R2"])
            fold_mape.append(m["MAPE_%"])

        label = "log1p(price)" if use_log else "raw price"
        print(f"  [{label}]  R²={np.mean(fold_r2):.4f}  MAPE={np.mean(fold_mape):.2f}%")

    print("=" * 60 + "\n")


def run(compare_log: bool = False, log_transform: bool = LOG_TRANSFORM):
    """
    Execute the full pipeline.

    Parameters
    ----------
    compare_log   : bool – if True, run log-transform comparison first
    log_transform : bool – which target to use for the final model
    """
    print("\n" + "=" * 60)
    print("  Unified Tree-based Model + SHAP Analytics")
    print("=" * 60 + "\n")

    # ── Optional: compare log vs raw ───────────────────────────
    if compare_log:
        compare_log_transforms()

    # ── Step 1-3: Train ────────────────────────────────────────
    results = train(data_path=DATA_PATH, log_transform=log_transform)

    best_name  = results["best_model_name"]
    pipeline   = results["best_pipeline"]
    feat_names = results["feature_names"]
    X_train    = results["X_train"]
    X_test     = results["X_test"]
    y_test     = results["y_test"]

    # ── Step 4: Evaluate ───────────────────────────────────────
    print(f"\n[Pipeline] Evaluating {best_name} on test set …")
    test_metrics = evaluate_and_plot(
        pipeline, X_test, y_test,
        log_transform=log_transform,
        model_name=best_name,
    )

    # ── Step 5: SHAP ───────────────────────────────────────────
    print(f"\n[Pipeline] Computing SHAP values for {best_name} …")
    analyzer = SHAPAnalyzer(pipeline, feat_names, log_transform=log_transform)

    # Compute on test set (use a subsample if test set is large)
    shap_sample = X_test.iloc[:min(500, len(X_test))]   # cap at 500 for speed
    shap_vals   = analyzer.compute_shap_values(shap_sample)

    # ── Step 6: Plots ──────────────────────────────────────────
    print("[Pipeline] Generating SHAP plots …")

    analyzer.plot_group_importance_summary(shap_vals,
        save_name="shap_group_importance_summary.png")

    analyzer.plot_group_bar(shap_vals, shap_sample,
        save_name="shap_group_bar.png")

    analyzer.plot_beeswarm(shap_vals, group="intrinsic",
        save_name="shap_beeswarm_intrinsic.png")

    analyzer.plot_beeswarm(shap_vals, group="extrinsic",
        save_name="shap_beeswarm_extrinsic.png")

    analyzer.plot_group_scatter(shap_vals, shap_sample,
        save_name="shap_group_scatter.png")

    # Waterfall for 3 representative samples: index 0, 50, 100
    for idx in [0, 50, 100]:
        if idx < len(shap_sample):
            analyzer.plot_waterfall(
                shap_vals, shap_sample, sample_idx=idx,
                title=f"SHAP Waterfall – Sample #{idx} ({best_name})",
                save_name=f"shap_waterfall_sample{idx}.png",
            )

    # ── Step 7: Sample decompositions (console output) ────────
    print("\n[Pipeline] Sample price decompositions:")
    for idx in [0, 50, 100]:
        if idx < len(shap_sample):
            analyzer.print_sample_decomposition(shap_vals, shap_sample, sample_idx=idx)

    # ── Step 8: Save decomposition DataFrame ──────────────────
    decomp_df = analyzer.decompose_dataset(shap_vals, shap_sample)
    decomp_path = os.path.join(OUTPUT_DIR, "shap_decomposition_test.csv")
    decomp_df.to_csv(decomp_path, index=False)
    print(f"[Pipeline] Decomposition table saved → {decomp_path}")

    # ── Done ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline completed successfully! 🎉")
    print(f"  Best model     : {best_name}")
    print(f"  Log transform  : {log_transform}")
    print(f"  R²  (test)     : {test_metrics['R2']:.4f}")
    print(f"  MAPE (test)    : {test_metrics['MAPE_%']:.2f}%")
    print(f"  Outputs saved in: {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    return results, analyzer, shap_vals


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Unified Tree-based Model + SHAP pipeline."
    )
    parser.add_argument(
        "--compare-log-transform",
        action="store_true",
        help="Compare CV performance with and without log1p target transform.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Use raw price as target (default: log1p transform).",
    )
    args = parser.parse_args()

    use_log = not args.no_log    # default True unless --no-log flag

    run(compare_log=args.compare_log_transform, log_transform=use_log)
