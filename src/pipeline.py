import os
import pandas as pd
from src.data_gen import generate_dataset
from src.preprocess import preprocess
from src.feature_store import FeatureStore


def run_pipeline(versions=(1, 2)):
    fs = FeatureStore()
    os.makedirs("data/raw", exist_ok=True)
    for v in versions:
        path = f"data/raw/battery_data_v{v}.csv"
        if os.path.exists(path):                      # use the provided dataset if present
            raw = pd.read_csv(path)
            src = "loaded provided CSV"
        else:
            raw = generate_dataset(n=250, version=v)
            raw.to_csv(path, index=False)
            src = "generated"
        feats = preprocess(raw)
        fs.write("battery_features", v, feats)        # feature store
        print(f"[pipeline] v{v}: raw {src}, {len(feats)} engineered rows stored in feature store")


if __name__ == "__main__":
    run_pipeline()
