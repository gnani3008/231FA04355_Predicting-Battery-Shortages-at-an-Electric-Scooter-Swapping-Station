import json
import os
from datetime import datetime
import pandas as pd

STORE = "data/feature_store"


class FeatureStore:
    def __init__(self, path=STORE):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.meta_file = os.path.join(path, "registry.json")

    def _meta(self):
        if os.path.exists(self.meta_file):
            with open(self.meta_file) as f:
                return json.load(f)
        return {}

    def _save(self, meta):
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def write(self, name, version, df):
        key = f"{name}_v{version}"
        df.to_csv(os.path.join(self.path, key + ".csv"), index=False)
        meta = self._meta()
        meta[key] = {"rows": len(df), "columns": list(df.columns),
                     "created": datetime.now().isoformat(timespec="seconds"), "reads": 0}
        self._save(meta)

    def read(self, name, version):
        key = f"{name}_v{version}"
        df = pd.read_csv(os.path.join(self.path, key + ".csv"))
        meta = self._meta()
        meta[key]["reads"] += 1          # tracks feature reuse
        self._save(meta)
        return df
