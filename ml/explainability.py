"""
VidyutDrishti — SHAP Explainability
Computes SHAP values for each prediction, showing which features
drive the forecast up or down. Used by the LLM for natural language explanations.
"""

import pandas as pd
import numpy as np
import shap
import joblib
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

HORIZONS = [1, 3, 6, 12, 24]


def compute_shap_values(node_type, horizon=1, num_background=100):
    """Compute SHAP values for test set predictions."""
    print(f"\n  Computing SHAP for {node_type} {horizon}h horizon...")

    feature_path = os.path.join(DATA_DIR, f"{node_type}_features.parquet")
    df = pd.read_parquet(feature_path)

    feature_cols = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_features.joblib"))
    model = joblib.load(os.path.join(MODEL_DIR, f"{node_type}_{horizon}h.joblib"))

    n = len(df)
    val_end = int(n * 0.85)

    X_test = df[feature_cols].iloc[val_end:]

    # Use TreeExplainer (fast for XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return shap_values, X_test, feature_cols, explainer.expected_value


def get_top_features(shap_values, feature_cols, sample_idx, top_n=5):
    """Get top N features driving a specific prediction."""
    sv = shap_values[sample_idx]
    feature_importance = list(zip(feature_cols, sv))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_importance[:top_n]


def explain_prediction(node_type, horizon, sample_idx, shap_values, X_test, feature_cols, base_value):
    """Generate structured explanation for a single prediction."""
    top_features = get_top_features(shap_values, feature_cols, sample_idx)

    prediction = base_value + shap_values[sample_idx].sum()
    prediction = max(0, prediction)

    explanation = {
        "node_type": node_type,
        "horizon_hours": horizon,
        "predicted_power_w": round(float(prediction), 2),
        "base_value_w": round(float(base_value), 2),
        "top_drivers": [],
    }

    for feature_name, shap_val in top_features:
        feature_value = X_test.iloc[sample_idx][feature_name]
        explanation["top_drivers"].append({
            "feature": feature_name,
            "shap_value": round(float(shap_val), 3),
            "feature_value": round(float(feature_value), 2),
            "direction": "increases" if shap_val > 0 else "decreases",
            "impact_w": round(abs(float(shap_val)), 3),
        })

    return explanation


def compute_global_importance(shap_values, feature_cols):
    """Compute mean absolute SHAP value per feature (global importance)."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance = list(zip(feature_cols, mean_abs_shap))
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


if __name__ == "__main__":
    print("=" * 60)
    print("  VidyutDrishti — SHAP Explainability")
    print("=" * 60)

    for node_type in ["solar", "wind"]:
        print(f"\n{'─'*60}")
        print(f"  {node_type.upper()} NODE")
        print(f"{'─'*60}")

        shap_values, X_test, feature_cols, base_value = compute_shap_values(node_type, horizon=1)

        # Global feature importance
        importance = compute_global_importance(shap_values, feature_cols)
        print(f"\n  Global Feature Importance (top 10):")
        for feat, imp in importance[:10]:
            print(f"    {feat:<35} {imp:.4f} W")

        # Example explanations
        print(f"\n  Example Predictions with Explanations:")
        for idx in [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4]:
            explanation = explain_prediction(
                node_type, 1, idx, shap_values, X_test, feature_cols, base_value)
            print(f"\n    Sample {idx}:")
            print(f"      Predicted: {explanation['predicted_power_w']:.2f} W")
            print(f"      Top drivers:")
            for d in explanation["top_drivers"][:3]:
                print(f"        {d['feature']}: {d['direction']} by {d['impact_w']:.3f}W "
                      f"(value={d['feature_value']:.2f})")

        # Save SHAP values for dashboard
        shap_path = os.path.join(MODEL_DIR, f"{node_type}_shap_values.joblib")
        joblib.dump({
            "shap_values": shap_values,
            "feature_cols": feature_cols,
            "base_value": base_value,
        }, shap_path)
        print(f"\n  Saved: {shap_path}")

    print("\n" + "=" * 60)
    print("  SHAP explainability computed for all models")
    print("=" * 60)
