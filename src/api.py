import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.preprocess import preprocess, FEATURES

app = FastAPI(title="Battery Shortage Predictor")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load("models/best_model.joblib")
    return _model


class StationConditions(BaseModel):
    charged_batteries_5pm: int = Field(ge=0)
    batteries_finishing_by_7pm: int = Field(ge=0)
    advance_bookings: int = Field(ge=0)
    historical_walkin_demand: int = Field(ge=0)
    rainfall_forecast_mm: float = Field(ge=0)
    is_weekend: int = Field(ge=0, le=1)
    nearby_event: int = Field(ge=0, le=1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: StationConditions):
    df = preprocess(pd.DataFrame([c.model_dump()]))
    X = df[FEATURES]
    model = get_model()
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": "Shortage Expected" if pred == 1 else "Sufficient Batteries",
        "shortage_probability": round(prob, 3),
    }
