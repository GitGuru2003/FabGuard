import numpy as np

from FabGuard.evaluation import compute_metrics, select_threshold

Y = np.array([0, 0, 0, 0, 1, 1])
SCORES = np.array([0.1, 0.2, 0.3, 0.6, 0.5, 0.9])


def test_compute_metrics_counts():
    m = compute_metrics(Y, SCORES, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 3)
    assert m["recall"] == 1.0
    assert np.isclose(m["precision"], 2 / 3)


def test_constant_scores_are_chance_level():
    m = compute_metrics(Y, np.zeros(len(Y)))
    assert m["roc_auc"] == 0.5
    assert np.isclose(m["pr_auc"], Y.mean())
    assert m["recall"] == 0.0


def test_select_threshold_max_f1_and_min_recall():
    assert select_threshold(Y, SCORES) == 0.5
    assert select_threshold(Y, SCORES, min_recall=0.5) == 0.9


def test_gm_and_mcc():
    # Park et al. (2024) Table 2: 1000 pass all correct, 5 of 10 failures caught -> GM 0.707
    y = np.array([0] * 1000 + [1] * 10)
    score = np.array([0.0] * 1000 + [1.0] * 5 + [0.0] * 5)
    m = compute_metrics(y, score)
    assert np.isclose(m["specificity"], 1.0)
    assert np.isclose(m["gm"], np.sqrt(0.5))
    assert 0 < m["mcc"] < 1
