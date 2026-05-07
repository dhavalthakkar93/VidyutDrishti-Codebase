"""
VidyutDrishti — Model Evaluation
Comprehensive evaluation against baselines with per-regime analysis.
Produces metrics table for submission/presentation.
"""

import pandas as pd
import numpy as np
import joblib
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

HORIZONS = [1, 3, 6, 12, 24]


def mape(y_true, y_pred):
    mask = y_true > 0.1
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def evaluate_node(node_type):
    """Full evaluation of a node's forecasting performance."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION: {node_type.upper()} NODE")
    print(f"{'='*70}")

    feature_path = os.path.join(DATA_DIR, f"{node_type}_features.parquet")
    df = pd.read_parquet(feature_path)

    feature_cols = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_features.joblib"))
    power_col = "power" if node_type == "solar" else "wind_power"

    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy()
    X_test = test_df[feature_cols].values
    hours = test_df["hour"].values if "hour" in test_df.columns else np.zeros(len(test_df))

    print(f"\n  Test set size: {len(test_df)} samples")
    print(f"  Test period: {test_df.index[0]} to {test_df.index[-1]}")

    # ============================================================
    # 1. Overall metrics per horizon
    # ============================================================
    print(f"\n  {'─'*65}")
    print(f"  {'Horizon':<10} {'MAE(W)':<10} {'RMSE(W)':<10} {'MAPE%':<10} "
          f"{'Persist MAE':<12} {'Improvement'}")
    print(f"  {'─'*65}")

    all_results = []

    for horizon in HORIZONS:
        target_col = f"target_{power_col}_{horizon}h"
        if target_col not in df.columns:
            continue

        model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib"))
        y_test = test_df[target_col].values
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        # Persistence: use current value as forecast
        y_persist = test_df[power_col].values

        mae_model = mae(y_test, y_pred)
        rmse_model = rmse(y_test, y_pred)
        mape_model = mape(y_test, y_pred)
        mae_persist = mae(y_test, y_persist)
        mape_persist = mape(y_test, y_persist)
        improvement = (1 - mae_model / mae_persist) * 100 if mae_persist > 0 else 0

        print(f"  {horizon}h{'':<7} {mae_model:<10.3f} {rmse_model:<10.3f} "
              f"{mape_model:<10.1f} {mae_persist:<12.3f} {improvement:.1f}%")

        all_results.append({
            "horizon": horizon,
            "mae": mae_model,
            "rmse": rmse_model,
            "mape": mape_model,
            "persist_mae": mae_persist,
            "persist_mape": mape_persist,
            "improvement_pct": improvement,
        })

    # ============================================================
    # 2. Day vs Night analysis (solar only)
    # ============================================================
    if node_type == "solar":
        print(f"\n  --- Day vs Night Analysis (1h horizon) ---")
        target_col = f"target_{power_col}_1h"
        model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_1h.joblib"))
        y_test = test_df[target_col].values
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        day_mask = (hours >= 7) & (hours <= 18)
        night_mask = ~day_mask

        if day_mask.sum() > 0:
            print(f"    Daytime (7-18h):  MAE={mae(y_test[day_mask], y_pred[day_mask]):.3f}W  "
                  f"MAPE={mape(y_test[day_mask], y_pred[day_mask]):.1f}%")
        if night_mask.sum() > 0:
            night_pred_nonzero = np.sum(y_pred[night_mask] > 0.01)
            print(f"    Nighttime:        MAE={mae(y_test[night_mask], y_pred[night_mask]):.3f}W  "
                  f"(pred>0 at night: {night_pred_nonzero} samples)")

    # ============================================================
    # 3. Conformal prediction coverage
    # ============================================================
    print(f"\n  --- Uncertainty Coverage ---")
    cal_path = os.path.join(MODEL_DIR, f"{node_type}_conformal.joblib")
    if os.path.exists(cal_path):
        conformal = joblib.load(cal_path)
        for horizon in [1, 6, 24]:
            target_col = f"target_{power_col}_{horizon}h"
            if target_col not in df.columns or horizon not in conformal:
                continue
            model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib"))
            y_test = test_df[target_col].values
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)

            for alpha in [0.80, 0.95]:
                q = conformal[horizon]["quantiles"].get(alpha, 0)
                lower = np.maximum(y_pred - q, 0)
                upper = y_pred + q
                coverage = np.mean((y_test >= lower) & (y_test <= upper))
                avg_width = np.mean(upper - lower)
                print(f"    {horizon}h | {int(alpha*100)}% CI: coverage={coverage*100:.1f}%  "
                      f"avg width={avg_width:.2f}W")
    else:
        print("    (run conformal_prediction.py first)")

    # ============================================================
    # 4. Ramp detection (>20% change in power over 1h)
    # ============================================================
    print(f"\n  --- Ramp Event Detection ---")
    target_1h = f"target_{power_col}_1h"
    if target_1h in df.columns:
        model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_1h.joblib"))
        y_test = test_df[target_1h].values
        y_current = test_df[power_col].values

        # Ramp = >20% change from current to 1h ahead
        with np.errstate(divide='ignore', invalid='ignore'):
            actual_change = np.abs(y_test - y_current) / np.maximum(y_current, 0.1)
        ramp_mask = actual_change > 0.20
        ramp_count = ramp_mask.sum()

        if ramp_count > 0:
            y_pred = model.predict(X_test)
            y_pred = np.maximum(y_pred, 0)
            ramp_mae = mae(y_test[ramp_mask], y_pred[ramp_mask])
            ramp_mape_val = mape(y_test[ramp_mask], y_pred[ramp_mask])
            print(f"    Ramp events (>20% change): {ramp_count} / {len(y_test)} samples")
            print(f"    During ramps: MAE={ramp_mae:.3f}W  MAPE={ramp_mape_val:.1f}%")
        else:
            print(f"    No ramp events detected in test set")

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("  VidyutDrishti — Comprehensive Model Evaluation")
    print("=" * 70)

    solar_results = evaluate_node("solar")
    wind_results = evaluate_node("wind")

    # Final summary table
    print(f"\n{'='*70}")
    print("  SUMMARY FOR SUBMISSION")
    print(f"{'='*70}")
    print(f"\n  {'Node':<8} {'1h MAPE':<10} {'6h MAPE':<10} {'24h MAPE':<10} {'Persist 1h':<12} {'Improvement'}")
    print(f"  {'─'*62}")

    for name, results in [("Solar", solar_results), ("Wind", wind_results)]:
        r1 = next((r for r in results if r["horizon"] == 1), None)
        r6 = next((r for r in results if r["horizon"] == 6), None)
        r24 = next((r for r in results if r["horizon"] == 24), None)
        if r1 and r6 and r24:
            print(f"  {name:<8} {r1['mape']:<10.1f} {r6['mape']:<10.1f} "
                  f"{r24['mape']:<10.1f} {r1['persist_mape']:<12.1f} {r1['improvement_pct']:.1f}%")
