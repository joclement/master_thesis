import pytest

from thesis import metrics


def test_file_score_valid_input():
    y_true = [0, 0, 0]
    predictions = [0, 2, 0]
    assert metrics.file_score(y_true, predictions) == 1.0

    predictions = [2, 2, 0]
    assert metrics.file_score(y_true, predictions) == 0.0

    y_true = [0, 0, 0]
    predictions = [0, 1, 2]
    assert metrics.file_score(y_true, predictions) == 1.0 / 3


def test_file_score_error():
    y_true = [0, 1]
    predictions = [0, 0]
    with pytest.raises(ValueError):
        metrics.file_score(y_true, predictions)
