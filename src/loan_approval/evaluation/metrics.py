"""Classification metrics for the default model."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: max separation between the score distributions of defaults and non-defaults."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def ks_threshold(y_true, y_score) -> tuple[float, float]:
    """Score cut-off where TPR - FPR is largest (the KS point), and the KS value there."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    i = int(np.argmax(tpr - fpr))
    return float(thresholds[i]), float(tpr[i] - fpr[i])


def confusion_at(y_true, y_score, threshold: float) -> dict[str, float]:
    """Confusion matrix and rates when loans with score >= threshold are predicted to default."""
    y, pred = np.asarray(y_true).astype(bool), np.asarray(y_score) >= threshold
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    fn, tn = int(np.sum(~pred & y)), int(np.sum(~pred & ~y))
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall": tp / (tp + fn),  # share of defaulters caught
        "false_positive_rate": fp / (fp + tn),  # share of good loans rejected
        "precision": tp / (tp + fp),
        "accuracy": (tp + tn) / len(y),
        "approval_rate": (tn + fn) / len(y),
        "default_rate_approved": fn / (tn + fn),
        "default_rate_rejected": tp / (tp + fp),
    }


def psi(expected, actual, bins: int = 10) -> float:
    """Population stability index of `actual` scores against `expected` (decile bins of expected)."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def classification_metrics(y_true, y_score) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "logloss": log_loss(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "default_rate": float(np.mean(y_true)),
        "mean_score": float(np.mean(y_score)),
    }
