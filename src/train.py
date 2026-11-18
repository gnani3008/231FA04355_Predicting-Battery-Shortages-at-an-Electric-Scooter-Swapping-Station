import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipeline import run_pipeline
from src.feature_store import FeatureStore
from src.preprocess import FEATURES, TARGET

MODEL_NAME = "battery_shortage_model"
TRUSTED = ["sklearn.tree._tree.Tree"]   # we trained this model ourselves, so it is safe


def log_and_register(model):
    """Works across MLflow versions (2.x and 3.x)."""
    attempts = [
        dict(name="model", registered_model_name=MODEL_NAME, skops_trusted_types=TRUSTED),
        dict(artifact_path="model", registered_model_name=MODEL_NAME, skops_trusted_types=TRUSTED),
        dict(name="model", registered_model_name=MODEL_NAME),
        dict(artifact_path="model", registered_model_name=MODEL_NAME),
    ]
    last = None
    for kw in attempts:
        try:
            mlflow.sklearn.log_model(model, **kw)
            return
        except TypeError as e:
            last = e
    raise last


def main():
    run_pipeline()
    fs = FeatureStore()
    # Feature reuse: training reads engineered features from the store (no recomputation)
    df = pd.concat([fs.read("battery_features", 1), fs.read("battery_features", 2)], ignore_index=True)
    print(f"[train] loaded {len(df)} rows from feature store (v1 + v2)")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        "GradientBoosting": Pipeline([("scaler", StandardScaler()),
                                      ("clf", GradientBoostingClassifier(random_state=42))]),
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("battery_shortage")

    results, cms = {}, {}
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            proba = pipe.predict_proba(X_te)[:, 1]
            m = {
                "accuracy": float(accuracy_score(y_te, pred)),
                "precision": float(precision_score(y_te, pred)),
                "recall": float(recall_score(y_te, pred)),
                "f1": float(f1_score(y_te, pred)),
                "roc_auc": float(roc_auc_score(y_te, proba)),
            }
            mlflow.log_param("model", name)
            mlflow.log_param("features", ",".join(FEATURES))
            mlflow.log_metrics(m)
            results[name] = m
            cms[name] = confusion_matrix(y_te, pred).tolist()
            print(name, {k: round(v, 3) for k, v in m.items()})
            print("confusion matrix:", cms[name])

    # best model: highest recall on shortage class (F1 as tie-breaker)
    best = max(results, key=lambda k: (round(results[k]["recall"], 4), results[k]["f1"]))
    print(f"\n[train] best model by recall: {best}")

    # save for FastAPI first, so serving never depends on MLflow
    joblib.dump(models[best], "models/best_model.joblib")
    with open("reports/results.json", "w") as f:
        json.dump({"best": best, "results": results, "confusion_matrices": cms,
                   "train_rows": len(X_tr), "test_rows": len(X_te)}, f, indent=2)

    with mlflow.start_run(run_name=f"register_{best}"):
        mlflow.log_param("model", best)
        mlflow.log_metrics(results[best])
        log_and_register(models[best])

    client = MlflowClient()
    latest = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "champion", str(latest))
    print(f"[mlflow] registered '{MODEL_NAME}' version {latest} with alias 'champion'")


if __name__ == "__main__":
    main()
