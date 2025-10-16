from __future__ import annotations
import dataclasses as dc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import pathlib as _p
import yaml

class ModelSpec(TypedDict, total=False):
    name: str
    type: str  # "logistic" | "hist_gbdt" | "rf"
    params: Dict[str, Any]

@dataclass
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 1
    shuffle: bool = True

@dataclass
class ValidationConfig:
    cv: CVConfig = CVConfig()

@dataclass
class DataConfig:
    kind: str = "synthetic"  # or "csv"
    path: Optional[str] = None
    target: str = "target"
    test_size: float = 0.2
    n_samples: int = 5000
    n_features: int = 20
    n_informative: int = 5
    n_redundant: int = 2
    n_repeated: int = 0
    n_classes: int = 2
    weights: Optional[List[float]] = dc.field(default_factory=lambda: [0.95, 0.05])
    n_clusters_per_class: int = 2

@dataclass
class CalibrationConfig:
    method: str = "sigmoid"  # or "isotonic"

@dataclass
class CostConfig:
    fn: float = 10.0
    fp: float = 1.0

@dataclass
class ReportsConfig:
    pr_k: Optional[int] = 100

@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"

@dataclass
class Config:
    random_state: int
    data: DataConfig
    validation: ValidationConfig
    models: List[ModelSpec]
    calibration: CalibrationConfig
    cost: CostConfig
    reports: ReportsConfig
    paths: PathsConfig

_SCHEMA_REQUIRED = {
    "random_state",
    "data",
    "validation",
    "models",
    "calibration",
    "cost",
    "reports",
    "paths",
}

def _validate_schema(d: Dict[str, Any]) -> None:
    missing = _SCHEMA_REQUIRED - set(d)
    if missing:
        raise ValueError(f"Config missing sections: {sorted(missing)}")
    if not isinstance(d["models"], list) or not d["models"]:
        raise ValueError("Config.models must be a non-empty list")

def load_config(path: str | _p.Path) -> Config:
    p = _p.Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _validate_schema(raw)

    def _dc_load(cls, sub: Dict[str, Any]):
        return cls(**sub) if isinstance(sub, dict) else cls()

    cfg = Config(
        random_state=int(raw.get("random_state", 42)),
        data=_dc_load(DataConfig, raw["data"]),
        validation=ValidationConfig(cv=_dc_load(CVConfig, raw["validation"].get("cv", {}))),
        models=raw["models"],
        calibration=_dc_load(CalibrationConfig, raw["calibration"]),
        cost=_dc_load(CostConfig, raw["cost"]),
        reports=_dc_load(ReportsConfig, raw["reports"]),
        paths=_dc_load(PathsConfig, raw["paths"]),
    )
    return cfg
