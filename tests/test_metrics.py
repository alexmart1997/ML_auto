import numpy as np

from sklearn.metrics import confusion_matrix, roc_auc_score


def test_roc_auc_is_perfect_for_ordered_scores():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert roc_auc_score(y_true, scores) == 1.0


def test_sklearn_confusion_matrix_counts_are_clear():
    y_true = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 1, 1])

    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    assert tp == 2
    assert fp == 1
    assert fn == 0
    assert tn == 1
