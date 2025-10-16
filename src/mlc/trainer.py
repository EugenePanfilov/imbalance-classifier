from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from .config import Config
from .data import make_dataset, train_test_split_stratified
from .features import build_preprocessor
from .models import build_model
from .validation import make_cv, oof_predict
from .calibration import calibrate
from .metrics import compute_metrics, bootstrap_ci
from .cost import expected_cost
from .plots import plot_pr_curve, plot_calibration, plot_cost_curve
from .persistence import save_artifacts
from .logging import setup_logging

logger = setup_logging(name="mlc.trainer")

def _aggregate(results):
    return { r["name"] : { "oof": r["oof"], "ci": r["ci"] } for r in results }

def run_training(cfg: Config) -> None:
    X, y = make_dataset(cfg)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(
        X, y, test_size=cfg.data.test_size, random_state=cfg.random_state, artifacts_dir=cfg.paths.artifacts_dir
    )

    preproc = build_preprocessor(X_tr, cfg)

    results = []
    for spec in cfg.models:
        model = build_model(spec, cfg.random_state)
        pipe = Pipeline([("preprocess", preproc), ("model", model)])
        cv = make_cv(cfg)
        oof_proba, oof_idx = oof_predict(pipe, X_tr, y_tr, cv)
        y_oof = y_tr.iloc[oof_idx].values
        oof_metrics = compute_metrics(y_oof, oof_proba, k=cfg.reports.pr_k)
        oof_ci = {
            "pr_auc_ci": bootstrap_ci(y_oof, oof_proba, scorer=average_precision_score, n_boot=400, seed=cfg.random_state),
            "roc_auc_ci": bootstrap_ci(y_oof, oof_proba, scorer=roc_auc_score, n_boot=400, seed=cfg.random_state),
            "brier_ci":   bootstrap_ci(y_oof, oof_proba, scorer=brier_score_loss, n_boot=400, seed=cfg.random_state),
        }
        results.append({
            "name": spec.get("name", spec.get("type")),
            "oof": oof_metrics,
            "ci": oof_ci,
            "pipe": pipe,
            "y_oof": y_oof,
            "oof_proba": oof_proba,
        })

    best = max(results, key=lambda r: r["oof"]["pr_auc"])
    logger.info("Selected best model: %s (PR-AUC=%.4f)", best["name"], best["oof"]["pr_auc"])

    # Honest plots on OOF predictions of the best model
    os.makedirs(cfg.paths.artifacts_dir, exist_ok=True)
    plot_pr_curve(best["y_oof"], best["oof_proba"], os.path.join(cfg.paths.artifacts_dir, "pr_curve.png"))
    plot_calibration(best["y_oof"], best["oof_proba"], os.path.join(cfg.paths.artifacts_dir, "calibration_curve.png"))

    # Calibrate best on full train
    cal = calibrate(best["pipe"], method=cfg.calibration.method, cv_or_holdout=5)
    cal.fit(X_tr, y_tr)

    # Cost-optimal threshold on train predictions after calibration
    proba_tr_cal = cal.predict_proba(X_tr)[:, 1]
    thr_grid = np.linspace(0.0, 1.0, 1001)
    best_thr, cost_curve = expected_cost(y_tr, proba_tr_cal, cfg.cost.fn, cfg.cost.fp, thr_grid)
    logger.info("Best threshold by cost: %.3f", best_thr)
    plot_cost_curve(cost_curve, os.path.join(cfg.paths.artifacts_dir, "cost_vs_threshold.png"))

    # Final evaluation on holdout test
    proba_test = cal.predict_proba(X_te)[:, 1]
    metrics_test = compute_metrics(y_te, proba_test, threshold=best_thr, k=cfg.reports.pr_k)

    save_artifacts(
        preproc=preproc,
        model=cal,
        metrics_cv=_aggregate(results),
        metrics_test=metrics_test,
        thresholds={"optimal": best_thr, "fixed_0_5": 0.5},
        paths=cfg.paths,
    )
