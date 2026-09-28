from FabGuard.data import load_secom, split_data


def test_load_secom_shapes_and_labels():
    X, y, timestamps = load_secom()
    assert X.shape == (1567, 590)
    assert len(y) == len(timestamps) == len(X)
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 104


def test_split_is_stratified():
    X, y, _ = load_secom()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) + len(X_test) == len(X)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
