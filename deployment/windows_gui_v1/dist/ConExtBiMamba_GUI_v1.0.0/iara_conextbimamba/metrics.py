from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, default=float("nan")):
    try:
        return float(fn())
    except Exception:
        return default


def binary_metrics(y_true, y_prob, threshold: float = 0.5, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        f"{prefix}accuracy": _safe_metric(lambda: accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": _safe_metric(lambda: balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}precision": _safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": _safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": _safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}auroc": _safe_metric(lambda: roc_auc_score(y_true, y_prob)),
        f"{prefix}auprc": _safe_metric(lambda: average_precision_score(y_true, y_prob)),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({
            f"{prefix}tn": float(tn),
            f"{prefix}fp": float(fp),
            f"{prefix}fn": float(fn),
            f"{prefix}tp": float(tp),
            f"{prefix}false_alarm_rate": float(fp / max(fp + tn, 1)),
            f"{prefix}miss_rate": float(fn / max(fn + tp, 1)),
        })
    except Exception:
        pass

    return metrics
