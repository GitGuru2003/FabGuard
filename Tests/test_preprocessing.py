import numpy as np
import pandas as pd

from FabGuard.preprocessing import CorrelationFilter, MissingnessFilter, build_preprocessor


def test_missingness_filter_uses_training_data_only():
    train = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    filt = MissingnessFilter(max_missing=0.5).fit(train)
    assert list(filt.get_feature_names_out()) == ["b"]
    assert filt.transform(test).shape == (2, 1)


def test_preprocessor_drops_missing_and_constant_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "mostly_missing": [np.nan] * 9 + [1.0],
            "constant": [5.0] * 10,
            "signal": rng.normal(size=10),
            "gappy": [np.nan, *rng.normal(size=9)],
        }
    )
    y = np.array([0, 1] * 5)
    Xt = build_preprocessor(n_features=None).fit_transform(X, y)
    assert Xt.shape == (10, 2)
    assert not np.isnan(Xt).any()
    np.testing.assert_allclose(Xt.mean(axis=0), 0, atol=1e-12)


def test_feature_selection_keeps_k_features():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 10)))
    y = np.array([0, 1] * 20)
    assert build_preprocessor(n_features=3).fit_transform(X, y).shape == (40, 3)


def test_correlation_filter_drops_redundant_column():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 50))
    X = pd.DataFrame({"a": a, "a_copy": a + 0.01 * rng.normal(size=50), "b": b})
    filt = CorrelationFilter(max_corr=0.8).fit(X)
    kept = list(filt.get_feature_names_out())
    assert len(kept) == 2 and "b" in kept
    assert CorrelationFilter(max_corr=1.0).fit(X).support_.all()
