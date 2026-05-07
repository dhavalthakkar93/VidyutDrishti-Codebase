"""
VidyutDrishti — Conformal Prediction (Uncertainty Quantification)
Calibrates prediction intervals using conformal prediction.
Provides guaranteed coverage without distributional assumptions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

HORIZONS = [1, 3, 6, 12, 24]
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95]


def calibrate_conformal(node_type):
    """
    Conformal prediction calibration.
    Uses the validation set residuals to compute prediction intervals.
    """
    print(f"\n{'='*60}")
    print(f"  Conformal Prediction — {node_type.upper()}")
    print(f"{'='*60}")

    feature_path = os.path.join(DATA_DIR, f"{node_type}_features.parquet")
    df = pd.read_parquet(feature_path)

    feature_cols = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_features.joblib"))
    power_col = "power" if node_type == "solar" else "wind_power"

    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    conformal_scores = {}

    for horizon in HORIZONS:
        target_col = f"target_{power_col}_{horizon}h"
        if target_col not in df.columns:
            continue

        model_path = os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib")
        model = joblib.load(model_path)

        # Use validation set for calibration
        X_val = df[feature_cols].iloc[train_end:val_end].values
        y_val = df[target_col].iloc[train_end:val_end].values

        y_pred_val = model.predict(X_val)
        y_pred_val = np.maximum(y_pred_val, 0)

        # Nonconformity scores = absolute residuals
        residuals = np.abs(y_val - y_pred_val)

        # Compute quantiles for each confidence level
        quantiles = {}
        for alpha in CONFIDENCE_LEVELS:
            q = np.quantile(residuals, alpha)
            quantiles[alpha] = q

        conformal_scores[horizon] = {
            "quantiles": quantiles,
            "residuals": residuals,
            "mean_residual": np.mean(residuals),
            "median_residual": np.median(residuals),
        }

        print(f"\n  Horizon {horizon}h:")
        print(f"    Mean |residual|: {np.mean(residuals):.3f} W")
        print(f"    Median |residual|: {np.median(residuals):.3f} W")
        for alpha, q in quantiles.items():
            print(f"    {int(alpha*100)}% CI half-width: ±{q:.3f} W")

    # Save calibration
    cal_path = os.path.join(MODEL_DIR, f"{node_type}_conformal.joblib")
    joblib.dump(conformal_scores, cal_path)
    print(f"\n  Saved: {cal_path}")

    # Verify coverage on test set
    print(f"\n  --- Coverage verification (test set) ---")
    for horizon in HORIZONS:
        target_col = f"target_{power_col}_{horizon}h"
        if target_col not in df.columns:
            continue

        model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib"))
        X_test = df[feature_cols].iloc[val_end:].values
        y_test = df[target_col].iloc[val_end:].values

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        for alpha in CONFIDENCE_LEVELS:
            q = conformal_scores[horizon]["quantiles"][alpha]
            lower = y_pred - q
            upper = y_pred + q
            lower = np.maximum(lower, 0)  # power >= 0
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            print(f"    {horizon}h | {int(alpha*100)}% CI: actual coverage = {coverage*100:.1f}%")

    return conformal_scores


def predict_with_uncertainty(model, X, conformal_scores, horizon, confidence=0.80):
    """Make prediction with confidence interval."""
    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)

    q = conformal_scores[horizon]["quantiles"][confidence]
    lower = np.maximum(y_pred - q, 0)
    upper = y_pred + q

    return y_pred, lower, upper


if __name__ == "__main__":
    solar_scores = calibrate_conformal("solar")
    wind_scores = calibrate_conformal("wind")

    print("\n" + "=" * 60)
    print("  Conformal prediction calibrated for all horizons")
    print("=" * 60)
