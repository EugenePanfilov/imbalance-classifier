from __future__ import annotations
import json
from pathlib import Path
from mlc.config import load_config
from mlc.trainer import run_training

def test_calibration_improves_brier_and_cost(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text((Path("configs/default.yaml").read_text()))
    cfg = load_config(cfg_path)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    run_training(cfg)

    art = Path(cfg.paths.artifacts_dir)
    metrics_test = json.loads((art / "metrics_test.json").read_text())
    assert "brier" in metrics_test

    thresholds = json.loads((art / "thresholds.json").read_text())
    assert thresholds["optimal"] != 0.5
