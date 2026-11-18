import numpy as np
import pandas as pd

RAW_FEATURES = [
    "charged_batteries_5pm", "batteries_finishing_by_7pm", "advance_bookings",
    "historical_walkin_demand", "rainfall_forecast_mm", "is_weekend", "nearby_event",
]
ENGINEERED = ["expected_battery_supply", "booking_pressure_ratio"]
FEATURES = RAW_FEATURES + ENGINEERED
NON_NEGATIVE = RAW_FEATURES[:5]
TARGET = "battery_shortage"


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df[RAW_FEATURES] = df[RAW_FEATURES].fillna(df[RAW_FEATURES].median())
    for c in NON_NEGATIVE:
        if (df[c] < 0).any():
            raise ValueError(f"Invalid negative values found in column '{c}'")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["expected_battery_supply"] = df["charged_batteries_5pm"] + df["batteries_finishing_by_7pm"]
    df["booking_pressure_ratio"] = df["advance_bookings"] / np.maximum(df["expected_battery_supply"], 1)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """validate -> impute -> engineer. Used by training AND serving."""
    return engineer_features(validate(df))
