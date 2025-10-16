from __future__ import annotations
from typing import Any, Mapping, Union
from .config import ModelSpec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def build_model(spec: Union[ModelSpec, Mapping[str, Any]], random_state: int):
    typ = spec.get("type")
    params = {**spec.get("params", {})}
    params.setdefault("random_state", random_state)

    if typ == "logistic":
        params.setdefault("max_iter", 200)
        return LogisticRegression(**params)

    if typ == "rf":
        return RandomForestClassifier(**params)

    if typ == "hist_gbdt":
        # у HistGradientBoostingClassifier нет class_weight
        return HistGradientBoostingClassifier(**{k: v for k, v in params.items() if k != "class_weight"})

    raise ValueError(f"Unknown model type: {typ}")