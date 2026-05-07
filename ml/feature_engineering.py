"""
VidyutDrishti — Feature Engineering
Transforms raw sensor data into ML-ready features with lag, rolling, and temporal encodings.
Produces hourly-aggregated data with forecast horizons (1h to 24h ahead).
"""

import sqlite3
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "training_data.db")
OUTPUT_PATH = os.path.join(DATA_DIR, "features.parquet")


def load_data(table):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.drop(columns=["id"], errors="ignore")
    return df


def resample_hourly(df):
    """Aggregate 1-minute data to hourly means."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].resample("1h").mean()


def add_temporal_features(df):
    """Add time-based features."""
    df["hour"] = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def add_lag_features(df, columns, lags):
    """Add lagged versions of specified columns."""
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(df, columns, windows):
    """Add rolling mean and std for specified columns."""
    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w).mean()
            df[f"{col}_roll_std_{w}h"] = df[col].rolling(w).std()
    return df


def add_diff_features(df, columns):
    """Add hour-over-hour change."""
    for col in columns:
        df[f"{col}_diff_1h"] = df[col].diff(1)
    return df


def create_target(df, target_col, horizons):
    """Create future target columns for multi-horizon forecasting."""
    for h in horizons:
        df[f"target_{target_col}_{h}h"] = df[target_col].shift(-h)
    return df


def build_solar_features():
    """Build feature set for solar power forecasting."""
    print("Loading solar data...")
    df = load_data("solar")
    df = resample_hourly(df)

    print("Adding features...")
    df = add_temporal_features(df)

    # Lag features: past values help predict future
    df = add_lag_features(df,
                          columns=["ghi", "power", "air_temp", "panel_temp", "humidity", "pressure", "soiling"],
                          lags=[1, 2, 3, 6, 12, 24])

    # Rolling statistics: capture trends
    df = add_rolling_features(df,
                              columns=["ghi", "power", "air_temp"],
                              windows=[3, 6, 12, 24])

    # Hour-over-hour changes: capture ramps
    df = add_diff_features(df, columns=["ghi", "power", "air_temp", "pressure"])

    # Clearness index: ratio of actual to theoretical clear-sky GHI
    # Approximation using max GHI per hour-of-day as clear sky reference
    hour_max_ghi = df.groupby("hour")["ghi"].transform("max")
    df["clearness_index"] = df["ghi"] / hour_max_ghi.replace(0, np.nan)
    df["clearness_index"] = df["clearness_index"].fillna(0)

    # Target: power output at various horizons ahead
    df = create_target(df, "power", horizons=[1, 2, 3, 6, 12, 24])

    df = df.dropna()
    return df


def build_wind_features():
    """Build feature set for wind power forecasting."""
    print("Loading wind data...")
    df = load_data("wind")
    df = resample_hourly(df)

    print("Adding features...")
    df = add_temporal_features(df)

    # Wind direction as sin/cos (circular encoding)
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_direction"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_direction"]))

    # Lag features
    df = add_lag_features(df,
                          columns=["wind_speed", "wind_power", "air_temp", "humidity", "pressure", "vibration"],
                          lags=[1, 2, 3, 6, 12, 24])

    # Rolling statistics
    df = add_rolling_features(df,
                              columns=["wind_speed", "wind_power", "vibration"],
                              windows=[3, 6, 12, 24])

    # Hour-over-hour changes
    df = add_diff_features(df, columns=["wind_speed", "wind_power", "pressure"])

    # Target: wind power at various horizons
    df = create_target(df, "wind_power", horizons=[1, 2, 3, 6, 12, 24])

    df = df.dropna()
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  VidyutDrishti — Feature Engineering")
    print("=" * 60)

    solar_df = build_solar_features()
    wind_df = build_wind_features()

    print(f"\nSolar features: {solar_df.shape[0]} samples x {solar_df.shape[1]} features")
    print(f"Wind features:  {wind_df.shape[0]} samples x {wind_df.shape[1]} features")

    # Save
    solar_path = os.path.join(DATA_DIR, "solar_features.parquet")
    wind_path = os.path.join(DATA_DIR, "wind_features.parquet")
    solar_df.to_parquet(solar_path)
    wind_df.to_parquet(wind_path)

    print(f"\nSaved: {solar_path}")
    print(f"Saved: {wind_path}")

    # Print feature list
    print(f"\n--- Solar feature columns ({len(solar_df.columns)}) ---")
    for col in sorted(solar_df.columns):
        print(f"  {col}")

    print(f"\n--- Wind feature columns ({len(wind_df.columns)}) ---")
    for col in sorted(wind_df.columns):
        print(f"  {col}")
