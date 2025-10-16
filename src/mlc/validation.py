from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone


def make_cv(cfg) -> RepeatedStratifiedKFold:
    cv_cfg = cfg.validation.cv
    return RepeatedStratifiedKFold(
        n_splits=cv_cfg.n_splits, n_repeats=cv_cfg.n_repeats, random_state=cfg.random_state
    )


def oof_predict(pipe, X: pd.DataFrame, y: pd.Series, cv) -> Tuple[np.ndarray, np.ndarray]:
    proba = np.zeros(len(y), dtype=float)
    oof_idx = np.zeros(len(y), dtype=bool)
    for tr_idx, va_idx in cv.split(X, y):
        model = clone(pipe)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = model.predict_proba(X.iloc[va_idx])[:, 1]
        proba[va_idx] = p
        oof_idx[va_idx] = True
    return proba[oof_idx], np.where(oof_idx)[0]
