from __future__ import annotations
import json
from pathlib import Path
from mlc.config import load_config
from mlc.trainer import run_training

def test_oof_prauc_and_artifacts(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text((Path("configs/default.yaml").read_text()))
    cfg = load_config(cfg_path)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    run_training(cfg)

    art = Path(cfg.paths.artifacts_dir)
    assert (art / "preprocessor.pkl").exists()
    assert (art / "model.pkl").exists()
    assert (art / "metrics_test.json").exists()
    d = json.loads((art / "metrics_cv.json").read_text())
    vals = [v["oof"]["pr_auc"] for v in d.values()]
    assert max(vals) > 0.2

def test_deterministic(tmp_path):
    from mlc.config import Config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text((Path("configs/default.yaml").read_text()))
    cfg = load_config(cfg_path)
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts1")
    run_training(cfg)

    cfg.paths.artifacts_dir = str(tmp_path / "artifacts2")
    run_training(cfg)

    import json as _json
    t1 = _json.loads((Path(tmp_path/"artifacts1"/"thresholds.json")).read_text())["optimal"]
    t2 = _json.loads((Path(tmp_path/"artifacts2"/"thresholds.json")).read_text())["optimal"]
    assert abs(t1 - t2) < 1e-6
