"""Model definitions compared in the project.

``build_pipeline`` chains preprocessing, optional SMOTE and a model into one
imblearn Pipeline, so preprocessing and resampling are refit inside every
cross-validation fold and never see validation or test data. ``model_specs``
returns each model with the small hyperparameter grid searched for it.
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from FabGuard import config
from FabGuard.preprocessing import build_preprocessor


class IsolationForestDetector(ClassifierMixin, BaseEstimator):
    """Isolation Forest wrapped as a classifier so it fits the same CV/scoring code.

    Training ignores ``y``. ``decision_function`` returns an anomaly score where
    higher means more anomalous (more likely a failure), so it can be ranked and
    thresholded like a classifier score.
    """

    def __init__(self, n_estimators=300, max_samples="auto", max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y=None):
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
        ).fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        return -self.forest_.score_samples(X)

    def predict(self, X):
        check_is_fitted(self)
        return (self.forest_.predict(X) == -1).astype(int)


def build_dummy():
    """Dummy classifier baseline: always predicts "pass", to show why accuracy alone is misleading."""
    return DummyClassifier(strategy="most_frequent")


def build_logistic_regression(class_weight: str | None = "balanced"):
    """L2-regularized logistic regression: interpretable supervised baseline."""
    return LogisticRegression(class_weight=class_weight, max_iter=5000, random_state=config.RANDOM_STATE)


def build_isolation_forest():
    """Isolation Forest: unsupervised; labels used only for evaluation/selection."""
    return IsolationForestDetector(random_state=config.RANDOM_STATE)


def build_xgboost(scale_pos_weight: float | None = None):
    """XGBoost: nonlinear supervised model. ``scale_pos_weight`` ~ n_pass / n_fail reweights failures."""
    return XGBClassifier(
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )


def build_pipeline(
    model,
    scale: bool = True,
    n_features: int | str | None = None,
    max_corr: float | None = None,
    smote: bool = False,
):
    """Chain preprocessing, optional SMOTE (training folds only) and ``model``."""
    # imblearn rejects nested Pipelines, so splice the preprocessing steps in directly
    steps = list(build_preprocessor(scale=scale, n_features=n_features, max_corr=max_corr).steps)
    if smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specs(y_train) -> dict:
    """Return ``{name: (pipeline, param_grid)}`` for every model in the comparison.

    Grids are kept deliberately small (see the overfitting risk in the proposal).
    """
    n_fail = int(y_train.sum())
    pos_weight = (len(y_train) - n_fail) / n_fail
    return {
        "dummy": (build_pipeline(build_dummy()), {}),
        "logreg": (
            build_pipeline(build_logistic_regression(), n_features="all", max_corr=1.0),
            {
                "corr__max_corr": [0.8, 1.0],
                "select__k": [20, 50, 100, "all"],
                "model__C": [0.0001, 0.001, 0.01, 0.1],
            },
        ),
        "isolation_forest": (
            build_pipeline(build_isolation_forest()),
            {"model__max_features": [0.5, 1.0], "model__max_samples": ["auto", 0.5]},
        ),
        "xgboost": (
            build_pipeline(build_xgboost(scale_pos_weight=pos_weight), scale=False, n_features="all"),
            {
                "select__k": [50, "all"],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.1],
            },
        ),
    }
