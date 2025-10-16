from __future__ import annotations
from typing import Callable, Dict, Optional, Tuple
import numpy as np
from sklearn import metrics as skm


def _f1_at_threshold(y_true, proba, thr: float) -> float:
    y_pred = (proba >= thr).astype(int)
    return skm.f1_score(y_true, y_pred)


def _recall_at_k(y_true, proba, k: Optional[int]) -> Optional[float]:
    if k is None or k <= 0:
        return None
    order = np.argsort(-proba)
    top_k = order[: min(k, len(order))]
    y_top = np.asarray(y_true)[top_k]
    positives = np.sum(y_true)
    if positives == 0:
        return 0.0
    return float(np.sum(y_top) / positives)


def compute_metrics(y_true, proba, threshold: float = 0.5, k: Optional[int] = None) -> Dict[str, float]:
    fpr, tpr, _ = skm.roc_curve(y_true, proba)
    precision, recall, _ = skm.precision_recall_curve(y_true, proba)
    metrics = {
        "roc_auc": skm.auc(fpr, tpr),
        "pr_auc": skm.auc(recall, precision),
        "brier": skm.brier_score_loss(y_true, proba),
        "accuracy": skm.accuracy_score(y_true, (proba >= threshold).astype(int)),
        "f1_at_thr": _f1_at_threshold(y_true, proba, threshold),
        "recall_at_k": _recall_at_k(y_true, proba, k) if k else None,
    }
    cm = skm.confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])
    metrics.update(
        {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    )
    return metrics


def bootstrap_ci(
    y_true, proba, scorer: Callable[[np.ndarray, np.ndarray], float], n_boot: int = 1000, seed: int = 0
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: list[float] = []
    idx = np.arange(n)
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        stats.append(float(scorer(y_true[s], proba[s])))
    # не перетираем тип списка массивом — используем отдельную переменную
    values = np.sort(np.array(stats, dtype=float))
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))