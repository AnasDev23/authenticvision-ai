"""Metrics and threshold calibration utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def calibrate_threshold(y_true, probabilities_ai, target_fpr: float = 0.05) -> float:
    """Choose a threshold from validation real-images to target a real-photo FPR.

    Call this only on validation data. The held-out test split must not be used
    to choose the threshold.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities_ai, dtype=np.float64)
    real_scores = p[y == 0]
    if real_scores.size == 0:
        return 0.5
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    try:
        threshold = np.quantile(real_scores, 1.0 - target_fpr, method="higher")
    except TypeError:  # NumPy < 1.22
        threshold = np.quantile(real_scores, 1.0 - target_fpr, interpolation="higher")
    return float(np.clip(threshold, 0.0, 1.0))


def calculate_metrics(y_true, probabilities_ai, threshold: float = 0.5) -> dict:
    """Calculate metrics where class 0=real and class 1=AI-generated."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities_ai, dtype=np.float64)
    if y.size == 0:
        return {"n_samples": 0, "threshold": float(threshold)}
    pred = (p >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    real_total = int(tn + fp)
    ai_total = int(tp + fn)
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
    ap = float(average_precision_score(y, p)) if np.unique(y).size == 2 else None
    return {
        "n_samples": int(y.size),
        "n_real": real_total,
        "n_ai": ai_total,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "f1_ai": float(f1_score(y, pred, zero_division=0)),
        "specificity_real": float(tn / real_total) if real_total else None,
        "false_positive_rate_real_as_ai": float(fp / real_total) if real_total else None,
        "false_negative_rate_ai_as_real": float(fn / ai_total) if ai_total else None,
        "roc_auc": auc,
        "average_precision": ap,
        "true_real": int(tn),
        "false_ai_alarm": int(fp),
        "missed_ai": int(fn),
        "true_ai": int(tp),
    }
