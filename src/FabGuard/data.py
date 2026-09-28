"""Loading and splitting the UCI SECOM dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

from FabGuard import config


def load_secom() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the SECOM sensor matrix, labels and timestamps.

    Returns
    -------
    X : DataFrame of shape (1567, 590), columns ``sensor_000`` ... ``sensor_589``.
    y : Series of 0 (pass) / 1 (fail). The raw file encodes pass as -1.
    timestamps : Series of datetimes aligned with ``X`` and ``y``.
    """
    X = pd.read_csv(config.SECOM_DATA_FILE, sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        config.SECOM_LABELS_FILE, sep=" ", header=None, names=["label", "timestamp"], quotechar='"'
    )
    y = (labels["label"] == 1).astype(int).rename("fail")
    timestamps = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S").rename("timestamp")

    return X, y, timestamps


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = config.TEST_SIZE):
    """Stratified train/test split. Do this before any learned preprocessing."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
    )
