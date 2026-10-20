"""Leakage-safe preprocessing. Everything here is fit on training data only.

Steps:
  1. Drop sensors with more than ``max_missing`` missing values
  2. Median imputation
  3. Drop zero-variance sensors
  4. Optionally drop redundant sensors (pairwise correlation above ``max_corr``)
  5. Standardization (for Logistic Regression; optional for tree models)
  6. Univariate feature selection (ANOVA F-test) to reduce dimensionality
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from FabGuard import config


class _ColumnMask(TransformerMixin, BaseEstimator):
    """Base for transformers that keep the columns flagged in ``support_`` during fit."""

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite="allow-nan", reset=False)
        return X[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        if input_features is None:
            input_features = getattr(
                self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)])
            )
        return np.asarray(input_features)[self.support_]


class MissingnessFilter(_ColumnMask):
    """Drop columns whose fraction of missing values in the training data exceeds ``max_missing``."""

    def __init__(self, max_missing: float = config.MAX_MISSING_FRACTION):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_all_finite="allow-nan")
        self.missing_fraction_ = np.isnan(X).mean(axis=0)
        self.support_ = self.missing_fraction_ <= self.max_missing
        return self


class CorrelationFilter(_ColumnMask):
    """Drop redundant columns until no pair has absolute correlation above ``max_corr``.

    Repeatedly takes the most correlated remaining pair and drops whichever of the
    two has the higher mean absolute correlation with the other remaining columns
    (the caret ``findCorrelation`` rule used by Munirathinam & Ramadoss, 2016).
    ``max_corr=1.0`` keeps everything, which is useful as a value in a grid.
    """

    def __init__(self, max_corr: float = 0.8):
        self.max_corr = max_corr

    def fit(self, X, y=None):
        X = validate_data(self, X)
        corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)), nan=0.0)
        np.fill_diagonal(corr, 0.0)
        keep = np.ones(X.shape[1], dtype=bool)
        while keep.sum() > 1:
            idx = np.flatnonzero(keep)
            sub = corr[np.ix_(idx, idx)]
            i, j = np.unravel_index(sub.argmax(), sub.shape)
            if sub[i, j] <= self.max_corr:
                break
            keep[idx[i] if sub[i].mean() > sub[j].mean() else idx[j]] = False
        self.support_ = keep
        return self


def build_preprocessor(
    max_missing: float = config.MAX_MISSING_FRACTION,
    scale: bool = True,
    n_features: int | str | None = None,
    max_corr: float | None = None,
) -> Pipeline:
    """Return an unfitted sklearn Pipeline implementing the steps above.

    ``n_features=None`` skips feature selection (required for unsupervised models,
    since SelectKBest uses the labels). Pass ``"all"`` to keep the step but select
    everything, which is useful as a value in a hyperparameter grid.
    ``max_corr=None`` skips the correlation filter.
    """
    steps = [
        ("missing", MissingnessFilter(max_missing=max_missing)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
    ]
    if max_corr is not None:
        steps.append(("corr", CorrelationFilter(max_corr=max_corr)))
    if scale:
        steps.append(("scale", StandardScaler()))
    if n_features is not None:
        steps.append(("select", SelectKBest(f_classif, k=n_features)))
    return Pipeline(steps)
