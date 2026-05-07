"""
VidyutDrishti — Prediction Pipeline
Loads trained models, generates forecasts with uncertainty and explanations.
Can be called from the dashboard or API server.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

HORIZONS = [1, 3, 6, 12, 24]


class VidyutPredictor:
    def __init__(self, node_type):
        self.node_type = node_type
        self.models = {}
        self.conformal = None
        self.feature_cols = None

        # Load models
        self.feature_cols = joblib.load(
            os.path.join(MODEL_DIR, f"{node_type}_features.joblib"))

        for h in HORIZONS:
            model_path = os.path.join(MODEL_DIR, f"{node_type}_{h}h.joblib")
            if os.path.exists(model_path):
                self.models[h] = joblib.load(model_path)

        # Load conformal calibration
        cal_path = os.path.join(MODEL_DIR, f"{node_type}_conformal.joblib")
        if os.path.exists(cal_path):
            self.conformal = joblib.load(cal_path)

    def predict(self, features_df, horizon=1, confidence=0.80):
        """
        Generate forecast with uncertainty.

        Args:
            features_df: DataFrame with feature columns matching training
            horizon: hours ahead (1, 3, 6, 12, or 24)
            confidence: confidence level (0.80, 0.90, or 0.95)

        Returns:
            dict with prediction, lower, upper bounds
        """
        if horizon not in self.models:
            raise ValueError(f"No model for horizon {horizon}h")

        X = features_df[self.feature_cols].values
        model = self.models[horizon]

        y_pred = model.predict(X)
        y_pred = np.maximum(y_pred, 0)

        # Uncertainty from conformal prediction
        lower = y_pred.copy()
        upper = y_pred.copy()
        if self.conformal and horizon in self.conformal:
            q = self.conformal[horizon]["quantiles"].get(confidence, 0)
            lower = np.maximum(y_pred - q, 0)
            upper = y_pred + q

        return {
            "prediction": y_pred,
            "lower": lower,
            "upper": upper,
            "confidence": confidence,
            "horizon_hours": horizon,
        }

    def predict_all_horizons(self, features_df, confidence=0.80):
        """Generate forecasts for all horizons."""
        results = {}
        for h in HORIZONS:
            if h in self.models:
                results[h] = self.predict(features_df, horizon=h, confidence=confidence)
        return results

    def explain(self, features_df, horizon=1, top_n=5):
        """Get SHAP-based explanation for prediction."""
        import shap

        model = self.models[horizon]
        X = features_df[self.feature_cols]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        explanations = []
        for i in range(len(X)):
            top_features = sorted(
                zip(self.feature_cols, shap_values[i]),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]

            prediction = max(0, explainer.expected_value + shap_values[i].sum())
            explanations.append({
                "predicted_power_w": round(float(prediction), 2),
                "top_drivers": [
                    {
                        "feature": name,
                        "impact_w": round(float(val), 3),
                        "direction": "+" if val > 0 else "-",
                        "value": round(float(X.iloc[i][name]), 2),
                    }
                    for name, val in top_features
                ]
            })

        return explanations


def forecast_next_24h(predictor, latest_features):
    """
    Generate a 24-hour forecast from current sensor state.
    Returns hourly predictions with confidence intervals.
    """
    results = predictor.predict_all_horizons(latest_features, confidence=0.80)

    forecast = []
    now = datetime.now()
    for h in sorted(results.keys()):
        r = results[h]
        forecast.append({
            "timestamp": (now + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"),
            "horizon_h": h,
            "power_w": round(float(r["prediction"][0]), 2),
            "lower_w": round(float(r["lower"][0]), 2),
            "upper_w": round(float(r["upper"][0]), 2),
            "confidence": r["confidence"],
        })

    return forecast


if __name__ == "__main__":
    from feature_engineering import build_solar_features, build_wind_features

    print("=" * 60)
    print("  VidyutDrishti — Prediction Demo")
    print("=" * 60)

    # Solar forecast
    print("\n--- Solar Forecast ---")
    solar_predictor = VidyutPredictor("solar")
    solar_df = build_solar_features()
    latest = solar_df.iloc[[-1]]  # last available sample

    forecast = forecast_next_24h(solar_predictor, latest)
    print(f"  {'Hour':<8} {'Power(W)':<12} {'80% CI'}")
    for f in forecast:
        print(f"  +{f['horizon_h']}h{'':<5} {f['power_w']:<12.2f} [{f['lower_w']:.2f}, {f['upper_w']:.2f}]")

    # Wind forecast
    print("\n--- Wind Forecast ---")
    wind_predictor = VidyutPredictor("wind")
    wind_df = build_wind_features()
    latest = wind_df.iloc[[-1]]

    forecast = forecast_next_24h(wind_predictor, latest)
    print(f"  {'Hour':<8} {'Power(W)':<12} {'80% CI'}")
    for f in forecast:
        print(f"  +{f['horizon_h']}h{'':<5} {f['power_w']:<12.2f} [{f['lower_w']:.2f}, {f['upper_w']:.2f}]")
