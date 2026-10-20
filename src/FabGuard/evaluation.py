"""Metrics, threshold selection and plots for imbalanced classification.

Primary metrics: precision, recall, F1, ROC-AUC and PR-AUC (emphasized).
Specificity, the geometric mean (GM) of sensitivity and specificity, and MCC are
included for comparison with prior SECOM work. Accuracy is reported only to show
how misleading it is on this dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from FabGuard.plotting import MODEL_COLORS, MODEL_NAMES, MUTED


def get_scores(model, X) -> np.ndarray:
    """Continuous failure score: P(fail) when available, else the decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Return threshold-dependent and threshold-free metrics for one model."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    constant = np.ptp(y_score) == 0
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "gm": np.sqrt(recall * specificity),
        "mcc": matthews_corrcoef(y_true, y_pred),
        # A constant score cannot rank samples: ROC-AUC is 0.5 and PR-AUC is the failure rate
        "roc_auc": 0.5 if constant else roc_auc_score(y_true, y_score),
        "pr_auc": y_true.mean() if constant else average_precision_score(y_true, y_score),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_threshold(y_true, y_score, min_recall: float | None = None) -> float:
    """Choose a decision threshold on validation (never test) data.

    With ``min_recall``, return the highest threshold whose recall is at least
    ``min_recall`` (fewest false alarms for the required detection rate).
    Otherwise return the threshold that maximizes F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more entry than thresholds; drop the final (recall=0) point
    precision, recall = precision[:-1], recall[:-1]
    if len(thresholds) == 0:
        return float(np.max(y_score))
    if min_recall is not None:
        ok = np.flatnonzero(recall >= min_recall)
        return float(thresholds[ok[-1]])
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return float(thresholds[np.argmax(f1)])


def plot_confusion_matrix(y_true, y_pred, ax=None, title: str | None = None):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=["pass", "fail"], cmap="Blues", colorbar=False, ax=ax
    )
    disp.ax_.grid(False)
    if title:
        disp.ax_.set_title(title)
    return disp.ax_


def plot_precision_recall_curves(results: dict, ax=None):
    """Overlay PR curves for several models: ``{name: (y_true, y_score)}``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_score) in results.items():
        # sklearn appends "(AP = ...)", i.e. PR-AUC, to the legend name
        PrecisionRecallDisplay.from_predictions(
            y_true, y_score, ax=ax, color=MODEL_COLORS.get(name), drawstyle="steps-post",
            name=MODEL_NAMES.get(name, name),
        )
    prevalence = np.mean(next(iter(results.values()))[0])
    ax.axhline(prevalence, color=MUTED, linestyle="--", linewidth=1, label=f"chance ({prevalence:.3f})")
    ax.set_xlabel("Recall (failures caught)")
    ax.set_ylabel("Precision (alarms that are real failures)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    return ax
