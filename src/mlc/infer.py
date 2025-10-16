from __future__ import annotations
import pandas as pd
from typing import Tuple
from .persistence import load_artifacts
from .config import PathsConfig

class InferenceModel:
    def __init__(self, preproc, model, threshold: float):
        self.preproc = preproc
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, paths: PathsConfig | dict) -> "InferenceModel":
        pre, model, thr = load_artifacts(paths)
        return cls(pre, model, thr)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        proba = self.model.predict_proba(df)[:, 1]
        label = (proba >= self.threshold).astype(int)
        return pd.Series(proba, name="proba"), pd.Series(label, name="label")
