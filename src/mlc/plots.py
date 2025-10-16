from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend to avoid Tk warnings in CI/tests
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y, proba, path: str) -> None:
    precision, recall, _ = precision_recall_curve(y, proba)
    plt.figure()
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_calibration(y, proba, path: str, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", linestyle="-", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_cost_curve(df_cost, path: str) -> None:
    plt.figure()
    plt.plot(df_cost["threshold"], df_cost["cost"])
    plt.xlabel("Threshold")
    plt.ylabel("Expected cost")
    plt.title("Cost vs Threshold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
