from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd

def expected_cost(y_true, proba, c_fn: float, c_fp: float, thresholds: np.ndarray) -> Tuple[float, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    costs = []
    for thr in thresholds:
        y_hat = (proba >= thr).astype(int)
        fp = int(((y == 0) & (y_hat == 1)).sum())
        fn = int(((y == 1) & (y_hat == 0)).sum())
        cost = c_fn * fn + c_fp * fp
        costs.append((thr, cost, fp, fn))
    df = pd.DataFrame(costs, columns=["threshold", "cost", "fp", "fn"]).sort_values("threshold").reset_index(drop=True)
    best_row = df.loc[df["cost"].idxmin()]
    return float(best_row["threshold"]), df
