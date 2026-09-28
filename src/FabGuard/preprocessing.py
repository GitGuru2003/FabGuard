"""Leakage-safe preprocessing. Everything here is fit on training data only.

Planned steps (proposal, Sep 28 - Oct 11):
  1. Drop sensors with more than ``MAX_MISSING_FRACTION`` missing values
  2. Median imputation
  3. Drop zero-variance sensors
  4. Standardization (for Logistic Regression; optional for tree models)
  5. Feature selection to reduce dimensionality
"""

from sklearn.pipeline import Pipeline

from FabGuard import config


def build_preprocessor(
    max_missing: float = config.MAX_MISSING_FRACTION,
    scale: bool = True,
    n_features: int | None = None,
) -> Pipeline:
    """Return an unfitted sklearn Pipeline implementing the steps above."""
    raise NotImplementedError
