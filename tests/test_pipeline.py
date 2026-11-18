import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data_gen import generate_dataset
from src.preprocess import preprocess
from src.api import app

client = TestClient(app)

GOOD = {"charged_batteries_5pm": 20, "batteries_finishing_by_7pm": 5, "advance_bookings": 15,
        "historical_walkin_demand": 30, "rainfall_forecast_mm": 5.0, "is_weekend": 1, "nearby_event": 1}


def test_dataset_size_and_target():
    df = generate_dataset(n=250, version=1)
    assert len(df) == 250 and set(df.battery_shortage.unique()) <= {0, 1}


def test_festival_version_has_more_shortages():
    assert generate_dataset(version=2).battery_shortage.mean() > generate_dataset(version=1).battery_shortage.mean()


def test_negative_battery_count_raises_error():
    df = generate_dataset(n=10)
    df.loc[0, "charged_batteries_5pm"] = -3
    with pytest.raises(ValueError):
        preprocess(df)


def test_engineered_features():
    out = preprocess(pd.DataFrame([GOOD]))
    assert out["expected_battery_supply"].iloc[0] == 25
    assert out["booking_pressure_ratio"].iloc[0] == pytest.approx(15 / 25)


def test_api_rejects_negative_batteries():
    bad = dict(GOOD, charged_batteries_5pm=-5)
    assert client.post("/predict", json=bad).status_code == 422


def test_api_prediction_label():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["prediction"] in ("Shortage Expected", "Sufficient Batteries")
