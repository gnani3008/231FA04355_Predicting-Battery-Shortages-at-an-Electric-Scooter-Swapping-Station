# Battery Shortage Predictor

Predicts whether an e-scooter battery-swapping station will run short of charged batteries between 5 PM and 7 PM.

- `src/data_gen.py` synthetic datasets (v1 normal, v2 college festival)
- `src/preprocess.py` validation and feature engineering
- `src/feature_store.py` and `src/pipeline.py` feature store and automated pipeline
- `src/train.py` Logistic Regression vs Gradient Boosting, MLflow registry
- `src/api.py` FastAPI service (`POST /predict`)
- `tests/` pytest suite, run by `.github/workflows/ci.yml`

Run: `pip install -r requirements.txt && python -m src.train && pytest -v && uvicorn src.api:app`
