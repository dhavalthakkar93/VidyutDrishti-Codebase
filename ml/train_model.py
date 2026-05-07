"""
VidyutDrishti — Model Training
Trains XGBoost models for solar and wind power forecasting.
Multi-horizon: 1h, 3h, 6h, 12h, 24h ahead.
Includes physics constraints and cross-validation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

HORIZONS = [1, 3, 6, 12, 24]


def mape(y_true, y_pred):
    mask = y_true > 0.1
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def persistence_baseline(df, target_col, horizon):
    """Persistence forecast: tomorrow = today (shift by 24h or by horizon)."""
    return df[target_col.replace(f"target_power_{horizon}h", "power")
              if "solar" in target_col else
              target_col.replace(f"target_wind_power_{horizon}h", "wind_power")]


def get_feature_columns(df, node_type):
    """Get input feature columns (exclude targets and raw power)."""
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if node_type == "solar":
        exclude = target_cols + ["power", "current"]
    else:
        exclude = target_cols + ["wind_power"]
    return [c for c in df.columns if c not in exclude]


def train_node(node_type):
    """Train models for a given node type (solar or wind)."""
    print(f"\n{'='*60}")
    print(f"  Training {node_type.upper()} forecasting models")
    print(f"{'='*60}")

    # Load features
    feature_path = os.path.join(DATA_DIR, f"{node_type}_features.parquet")
    df = pd.read_parquet(feature_path)
    print(f"  Dataset: {df.shape[0]} samples x {df.shape[1]} columns")

    power_col = "power" if node_type == "solar" else "wind_power"
    feature_cols = get_feature_columns(df, node_type)
    print(f"  Features: {len(feature_cols)}")

    results = {}

    for horizon in HORIZONS:
        target_col = f"target_{power_col}_{horizon}h"
        if target_col not in df.columns:
            print(f"  Skipping horizon {horizon}h — target not found")
            continue

        print(f"\n  --- Horizon: {horizon}h ahead ---")

        X = df[feature_cols].values
        y = df[target_col].values

        # Time series split: 70% train, 15% val, 15% test
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        # XGBoost parameters
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Predictions
        y_pred_test = model.predict(X_test)

        # Physics constraints: power cannot be negative
        y_pred_test = np.maximum(y_pred_test, 0)

        # Solar: power must be 0 at night (hour feature check)
        if node_type == "solar":
            hour_idx = feature_cols.index("hour") if "hour" in feature_cols else None
            if hour_idx is not None:
                night_mask = (X_test[:, hour_idx] < 6) | (X_test[:, hour_idx] > 19)
                y_pred_test[night_mask] = 0

        # Metrics
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape_val = mape(y_test, y_pred_test)

        # Persistence baseline (use value from 24h ago)
        y_persistence = df[power_col].iloc[val_end:].values
        y_persistence = np.maximum(y_persistence, 0)
        mae_persistence = mean_absolute_error(y_test, y_persistence[:len(y_test)])
        mape_persistence = mape(y_test, y_persistence[:len(y_test)])

        improvement = (1 - mae / mae_persistence) * 100 if mae_persistence > 0 else 0

        print(f"  Model — MAE: {mae:.3f} W | RMSE: {rmse:.3f} W | MAPE: {mape_val:.1f}%")
        print(f"  Persistence baseline — MAE: {mae_persistence:.3f} W | MAPE: {mape_persistence:.1f}%")
        print(f"  Improvement over persistence: {improvement:.1f}%")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib")
        joblib.dump(model, model_path)

        results[horizon] = {
            "mae": mae, "rmse": rmse, "mape": mape_val,
            "baseline_mae": mae_persistence, "baseline_mape": mape_persistence,
            "improvement": improvement,
        }

    # Save feature list
    feature_list_path = os.path.join(MODEL_DIR, f"{node_type}_features.joblib")
    joblib.dump(feature_cols, feature_list_path)

    # Summary
    print(f"\n  {'─'*50}")
    print(f"  {node_type.upper()} FORECAST SUMMARY")
    print(f"  {'─'*50}")
    print(f"  {'Horizon':<10} {'MAE(W)':<10} {'MAPE%':<10} {'Baseline%':<12} {'Improvement'}")
    for h, r in results.items():
        print(f"  {h}h{'':<7} {r['mae']:<10.3f} {r['mape']:<10.1f} {r['baseline_mape']:<12.1f} {r['improvement']:.1f}%")

    return results


if __name__ == "__main__":
    solar_results = train_node("solar")
    wind_results = train_node("wind")

    print("\n" + "=" * 60)
    print("  All models trained and saved to:", MODEL_DIR)
    print("=" * 60)
    print("\nModel files:")
    for f in sorted(os.listdir(MODEL_DIR)):
        size = os.path.getsize(os.path.join(MODEL_DIR, f))
        print(f"  {f} ({size / 1024:.0f} KB)")
