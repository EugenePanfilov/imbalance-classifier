from __future__ import annotations
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def calibrate(estimator, method: str, cv_or_holdout=5):
    method = method.lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'")
    # sklearn >= 1.4 uses 'estimator' instead of deprecated 'base_estimator'
    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv_or_holdout)

def calibration_curves(y_true, proba, n_bins: int = 10):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos
