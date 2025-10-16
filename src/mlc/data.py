from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from .config import Config
from .logging import setup_logging

logger = setup_logging(name="mlc.data")

def make_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    if cfg.data.kind == "synthetic":
        X, y = make_classification(
            n_samples=cfg.data.n_samples,
            n_features=cfg.data.n_features,
            n_informative=cfg.data.n_informative,
            n_redundant=cfg.data.n_redundant,
            n_repeated=cfg.data.n_repeated,
            n_classes=cfg.data.n_classes,
            weights=cfg.data.weights,
            n_clusters_per_class=cfg.data.n_clusters_per_class,
            random_state=cfg.random_state,
        )
        cols = [f"x{i:02d}" for i in range(cfg.data.n_features)]
        X = pd.DataFrame(X, columns=cols)
        y = pd.Series(y, name=cfg.data.target)
        X["cat_bin"] = pd.qcut(X[cols[0]], q=4, labels=["a", "b", "c", "d"]).astype(str)
        logger.info("Synthetic dataset created: %s rows, %s cols", X.shape[0], X.shape[1])
        return X, y
    elif cfg.data.kind == "csv" and cfg.data.path:
        df = pd.read_csv(cfg.data.path)
        y = df[cfg.data.target]
        X = df.drop(columns=[cfg.data.target])
        logger.info("CSV dataset loaded: %s rows, %s cols", X.shape[0], X.shape[1])
        return X, y
    else:
        raise ValueError("Unsupported data.kind; use 'synthetic' or provide csv path")

def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int, artifacts_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    os.makedirs(artifacts_dir, exist_ok=True)
    test_df = X_te.copy()
    test_df[y.name] = y_te.values
    test_path = os.path.join(artifacts_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    logger.info("Saved holdout test set to %s", test_path)
    return X_tr, X_te, y_tr, y_te
