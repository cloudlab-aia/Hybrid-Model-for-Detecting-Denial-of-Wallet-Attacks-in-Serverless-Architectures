"""
dow/metrics.py
==============
Evaluation helpers and all console-output / reporting functions.

Public API
----------
evaluate(y_true, y_pred, y_prob)  -> (metrics_dict, confusion_matrix)
print_model_report(name, m, elapsed)
print_confusion(cm)
print_classification_report(y_true, y_pred)
print_summary(results)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Console width
_W = 74


# Core evaluation
def _tpr_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float,
) -> float:
    """Return the True Positive Rate when FPR is closest to *target_fpr*."""
    fpr_v, tpr_v, _ = roc_curve(y_true, y_prob)
    return float(tpr_v[np.argmin(np.abs(fpr_v - target_fpr))])


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> tuple:
    """
    Compute all evaluation metrics

    Returns
    -------
    metrics : dict  {metric_name: float}
    cm      : 2×2 np.ndarray  confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    metrics = dict(
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall    = recall_score(y_true, y_pred, zero_division=0),
        F1        = f1_score(y_true, y_pred, zero_division=0),
        Accuracy  = accuracy_score(y_true, y_pred),
        ROC_AUC   = roc_auc_score(y_true, y_prob),
        PR_AUC    = average_precision_score(y_true, y_prob),
        TPR_1FPR  = _tpr_at_fpr(y_true, y_prob, 0.01),
        TPR_5FPR  = _tpr_at_fpr(y_true, y_prob, 0.05),
    )
    return metrics, cm


# Display helpers
def _sep(char: str = "=") -> None:
    print(char * _W)


def print_model_report(
    name:    str,
    metrics: dict,
    elapsed: float,
) -> None:
    """Print a formatted per-model metrics block"""
    _sep()
    print(f"  MODEL : {name.upper()}")
    _sep("-")
    print(f"  {'Metric':<22} {'Value':>10}")
    _sep("-")

    rows = [
        ("Precision",  "Precision"),
        ("Recall",     "Recall"),
        ("F1-score",   "F1"),
        ("Accuracy",   "Accuracy"),
        ("ROC-AUC",    "ROC_AUC"),
        ("PR-AUC",     "PR_AUC"),
        ("TPR@1%FPR",  "TPR_1FPR"),
        ("TPR@5%FPR",  "TPR_5FPR"),
    ]
    for label, key in rows:
        val = metrics.get(key, float("nan"))
        print(f"  {label:<22} {val:>10.4f}")

    _sep("-")
    print(f"  Training / inference time : {elapsed:.2f}s")
    _sep()


def print_confusion(cm: np.ndarray) -> None:
    """Print a labelled 2×2 confusion matrix."""
    _sep("-")
    print("  Confusion Matrix")
    print(f"  {'':<16} {'Pred Benign':>14} {'Pred Attack':>14}")
    print(f"  {'Actual Benign':<16} {cm[0, 0]:>14,} {cm[0, 1]:>14,}")
    print(f"  {'Actual Attack':<16} {cm[1, 0]:>14,} {cm[1, 1]:>14,}")
    _sep("-")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Print sklearn's full classification report."""
    print(classification_report(
        y_true, y_pred,
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))


def print_summary(results: dict) -> None:
    """
    Print a condensed results table for all evaluated models.
    Only shown when more than one model has been evaluated.
    """
    if len(results) < 2:
        return

    _sep()
    print("  FULL RESULTS SUMMARY")
    _sep("-")
    hdr = (
        f"  {'Model':<20} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} "
        f"{'Acc':>6} {'ROC':>6} {'PR':>6} "
        f"{'T@1%':>6} {'T@5%':>6}"
    )
    print(hdr)
    _sep("-")

    for name, m in results.items():
        print(
            f"  {name:<20} "
            f"{m['Precision']:>6.4f} {m['Recall']:>6.4f} {m['F1']:>6.4f} "
            f"{m['Accuracy']:>6.4f} {m['ROC_AUC']:>6.4f} {m['PR_AUC']:>6.4f} "
            f"{m['TPR_1FPR']:>6.4f} {m['TPR_5FPR']:>6.4f}"
        )

    _sep()
