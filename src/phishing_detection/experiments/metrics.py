"""Research metrics used by the exploratory paper experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def fpr_at_recall(y_true, y_scores, target_recall: float = 0.98) -> float | None:
    """Return the lowest FPR available at or above the target recall."""
    if y_scores is None:
        return None

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    if len(thresholds) == 0:
        return None

    candidate_fprs = []
    for threshold, recall_value in zip(thresholds, recall[:-1]):
        if recall_value < target_recall:
            continue
        y_pred = (np.asarray(y_scores) >= threshold).astype(int)
        tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        candidate_fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)

    if not candidate_fprs:
        return None
    return float(min(candidate_fprs))


def calculate_research_metrics(y_true, y_pred, y_scores=None) -> dict:
    """Calculate metrics for imbalanced, operational phishing evaluation."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_negative_rate": float(tnr),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "fpr_at_recall_98": fpr_at_recall(y_true, y_scores, 0.98),
    }

    if y_scores is not None:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_scores))
    else:
        metrics["auc_pr"] = None

    return metrics


def best_threshold_for_recall(y_true, y_scores, target_recall: float = 0.98) -> dict:
    """Find the threshold with the lowest FPR while meeting target recall."""
    if y_scores is None:
        return {
            "threshold": None,
            "target_recall": target_recall,
            "recall": None,
            "false_positive_rate": None,
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best = None
    for threshold, recall_value, precision_value in zip(thresholds, recall[:-1], precision[:-1]):
        if recall_value < target_recall:
            continue
        y_pred = (np.asarray(y_scores) >= threshold).astype(int)
        tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        candidate = {
            "threshold": float(threshold),
            "target_recall": float(target_recall),
            "recall": float(recall_value),
            "precision": float(precision_value),
            "false_positive_rate": float(fpr),
        }
        if best is None or candidate["false_positive_rate"] < best["false_positive_rate"]:
            best = candidate

    return best or {
        "threshold": None,
        "target_recall": target_recall,
        "recall": None,
        "precision": None,
        "false_positive_rate": None,
    }


def metrics_at_threshold(y_true, y_scores, threshold: float, prefix: str = "") -> dict:
    """Calculate metrics using a score threshold."""
    y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    metrics = calculate_research_metrics(y_true, y_pred, y_scores)
    if not prefix:
        return metrics
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def simulate_base_rates(
    y_true,
    y_pred,
    base_rates: list[float] | None = None,
    emails_per_batch: int = 10_000,
) -> list[dict]:
    """Simulate FP/FN counts per batch under realistic phishing base rates."""
    base_rates = base_rates or [0.5, 0.1, 0.05, 0.01, 0.005]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    simulations = []
    for base_rate in base_rates:
        phishing_count = emails_per_batch * base_rate
        legitimate_count = emails_per_batch - phishing_count
        false_positives = legitimate_count * fpr
        false_negatives = phishing_count * fnr
        true_positives = phishing_count * recall
        simulations.append({
            "base_rate": float(base_rate),
            "emails": int(emails_per_batch),
            "expected_false_positives": float(false_positives),
            "expected_false_negatives": float(false_negatives),
            "expected_true_positives": float(true_positives),
            "flagged_percent": float((false_positives + true_positives) / emails_per_batch),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        })
    return simulations
