"""
VidyutDrishti — Full ML Pipeline Runner
Executes the complete pipeline: data generation → features → training → evaluation.
Run this single script to reproduce all results.
"""

import subprocess
import sys
import os
import time

SCRIPTS = [
    ("generate_synthetic_data.py", "Generating 30 days synthetic data (Nadiad, Gujarat)"),
    ("feature_engineering.py", "Building ML features (lag, rolling, temporal)"),
    ("train_model.py", "Training XGBoost models (5 horizons x 2 nodes)"),
    ("conformal_prediction.py", "Calibrating uncertainty intervals"),
    ("explainability.py", "Computing SHAP explainability"),
    ("evaluate.py", "Running comprehensive evaluation"),
]


def run_script(script, description):
    print(f"\n{'━'*70}")
    print(f"  STEP: {description}")
    print(f"  Script: {script}")
    print(f"{'━'*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  ERROR: {script} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n  Completed in {elapsed:.1f}s")
    return elapsed


if __name__ == "__main__":
    print("=" * 70)
    print("  VidyutDrishti — Complete ML Pipeline")
    print("  Location: Nadiad, Gujarat (22.69N, 72.86E)")
    print("  Forecasting: Solar + Wind power, 1-24h ahead")
    print("=" * 70)

    total_time = 0
    for script, desc in SCRIPTS:
        total_time += run_script(script, desc)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"\n  Outputs:")
    print(f"    data/training_data.db     — Raw synthetic sensor data (30 days)")
    print(f"    data/solar_features.parquet — Solar ML features")
    print(f"    data/wind_features.parquet  — Wind ML features")
    print(f"    models/*.joblib            — Trained models + calibration")
    print(f"\n  Next: Run the dashboard with `python3 dashboard.py`")
