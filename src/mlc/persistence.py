from __future__ import annotations
import os
import json
from typing import Any, Dict, Tuple
import joblib


def save_artifacts(
    preproc,
    model,
    metrics_cv: Dict[str, Any],
    metrics_test: Dict[str, Any],
    thresholds: Dict[str, Any],
    paths,
) -> None:
    art = paths.artifacts_dir
    os.makedirs(art, exist_ok=True)
    joblib.dump(preproc, os.path.join(art, "preprocessor.pkl"))
    joblib.dump(model, os.path.join(art, "model.pkl"))
    with open(os.path.join(art, "metrics_cv.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_cv, f, indent=2)
    with open(os.path.join(art, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_test, f, indent=2)
    with open(os.path.join(art, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


def load_artifacts(paths) -> Tuple[Any, Any, float]:
    art = paths.artifacts_dir if hasattr(paths, "artifacts_dir") else paths["artifacts_dir"]
    pre = joblib.load(os.path.join(art, "preprocessor.pkl"))
    model = joblib.load(os.path.join(art, "model.pkl"))
    with open(os.path.join(art, "thresholds.json"), "r", encoding="utf-8") as f:
        thr = json.load(f).get("optimal", 0.5)
    return pre, model, float(thr)
