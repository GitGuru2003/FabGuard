"""Model definitions compared in the project.

Each builder returns an unfitted estimator (or imblearn Pipeline) so that
preprocessing and any resampling (e.g. SMOTE) run inside cross-validation folds
and never touch validation/test data.
"""


def build_dummy():
    """Dummy classifier baseline: shows why accuracy alone is misleading."""
    raise NotImplementedError


def build_logistic_regression(class_weight: str | None = "balanced"):
    """Regularized logistic regression: interpretable supervised baseline."""
    raise NotImplementedError


def build_isolation_forest():
    """Isolation Forest: unsupervised; labels used only for evaluation/selection."""
    raise NotImplementedError


def build_xgboost(scale_pos_weight: float | None = None):
    """XGBoost: nonlinear supervised model."""
    raise NotImplementedError
