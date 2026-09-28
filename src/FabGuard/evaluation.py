"""Metrics, threshold selection and plots for imbalanced classification.

Primary metrics: precision, recall, F1, ROC-AUC and PR-AUC (emphasized).
"""


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Return precision, recall, F1, ROC-AUC and PR-AUC for one model."""
    raise NotImplementedError


def select_threshold(y_true, y_score, min_recall: float | None = None) -> float:
    """Choose a decision threshold on validation (never test) data."""
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, ax=None):
    raise NotImplementedError


def plot_precision_recall_curves(results: dict, ax=None):
    """Overlay PR curves for several models: ``{name: (y_true, y_score)}``."""
    raise NotImplementedError
