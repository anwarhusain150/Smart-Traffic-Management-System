"""
data_preprocessing.py
---------------------
Reusable functions to LOAD, CLEAN and add FEATURES to the
UCI Metro Interstate Traffic Volume dataset.

Notebooks, the training script and the Streamlit app all import these
functions, so the cleaning is written only once (no copy-paste bugs).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Project root = the folder above "src"
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "Dataset" / "Metro_Interstate_Traffic_Volume.csv.gz"
CLEAN_PATH = ROOT / "Dataset" / "traffic_clean.csv"

# Congestion thresholds (vehicles per hour) used to create the class label
LOW_MAX = 2000      # volume < 2000           -> Low
MEDIUM_MAX = 4500   # 2000 <= volume < 4500   -> Medium, >= 4500 -> High
CLASS_ORDER = ["Low", "Medium", "High"]


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV (pandas can read .gz files directly)."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean the raw data. Each step is explained in a comment."""
    df = df.copy()
    log = print if verbose else (lambda *a, **k: None)
    log(f"Start: {df.shape[0]} rows")

    # 1) Datatype correction: date_time is text -> convert to real datetime
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 2) Exact duplicate rows -> remove
    n = len(df)
    df = df.drop_duplicates()
    log(f"Removed {n - len(df)} exact duplicate rows")

    # 3) Same hour appears many times (one row per weather condition,
    #    e.g. 'Rain' AND 'Mist'). Keep ONE row per hour so that every
    #    hour is counted only once.
    n = len(df)
    df = df.sort_values("date_time").drop_duplicates(subset="date_time", keep="first")
    log(f"Removed {n - len(df)} repeated timestamps (multiple weather rows for same hour)")

    # 4) Outliers that are physically impossible:
    #    temp = 0 Kelvin (-273 C) is a sensor error -> drop those rows
    n = len(df)
    df = df[df["temp"] > 0]
    log(f"Removed {n - len(df)} rows with temp = 0 K (sensor error)")
    #    rain_1h of ~9800 mm in one hour is impossible -> drop
    n = len(df)
    df = df[df["rain_1h"] < 100]
    log(f"Removed {n - len(df)} rows with impossible rainfall (>100 mm/h)")

    # 5) Holiday column: the name is only written on the 00:00 row of the
    #    holiday. We spread it to all 24 hours of that date.
    df["holiday"] = df["holiday"].replace("None", np.nan)
    holiday_dates = df.loc[df["holiday"].notna(), "date_time"].dt.date.unique()
    df["is_holiday"] = df["date_time"].dt.date.isin(holiday_dates).astype(int)

    # 6) Unit conversion: Kelvin -> Celsius (easier to understand)
    df["temp_c"] = (df["temp"] - 273.15).round(2)

    df = df.drop(columns=["holiday", "temp"]).reset_index(drop=True)
    log(f"End: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: create new useful columns from date_time."""
    df = df.copy()
    dt = df["date_time"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0 = Monday ... 6 = Sunday
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Rush hour on working days (7-9 AM and 4-6 PM)
    df["is_rush_hour"] = (
        df["hour"].isin([7, 8, 9, 16, 17, 18]) & (df["is_weekend"] == 0) & (df["is_holiday"] == 0)
    ).astype(int)
    # Cyclical encoding: hour 23 and hour 0 are neighbours, sin/cos captures that
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Combine rare weather groups into 'Other'
    rare = ["Smoke", "Squall"]
    df["weather_main"] = df["weather_main"].replace(rare, "Other")
    return df


def make_congestion_label(volume) -> pd.Series:
    """Convert traffic volume (number) to Low / Medium / High."""
    bins = [-np.inf, LOW_MAX, MEDIUM_MAX, np.inf]
    return pd.cut(volume, bins=bins, labels=CLASS_ORDER, right=False).astype(str)


# Columns used as model inputs (traffic_volume is the target, so it is NOT here)
NUMERIC_FEATURES = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "hour_sin", "hour_cos"]
CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "weather_main"]
BINARY_FEATURES = ["is_weekend", "is_holiday", "is_rush_hour"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def load_clean_featured() -> pd.DataFrame:
    """One call to get the final modelling table."""
    if CLEAN_PATH.exists():
        df = pd.read_csv(CLEAN_PATH, parse_dates=["date_time"])
    else:
        df = clean_data(load_raw(), verbose=False)
    df = add_features(df)
    df["congestion_level"] = make_congestion_label(df["traffic_volume"])
    return df


if __name__ == "__main__":
    clean = clean_data(load_raw())
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
