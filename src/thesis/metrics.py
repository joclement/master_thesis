from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils import check_array, check_consistent_length, column_or_1d
from sklearn.utils._encode import _encode, _unique
from sklearn.utils.multiclass import type_of_target

from . import constants
from .models import get_classifier, no_predict_proba


def avg_file_scores(
    y_true: pd.Series,
    proba_predictions: Union[pd.DataFrame, np.ndarray],
    sample_weight=None,
) -> float:
    proba_predictions = pd.DataFrame(
        proba_predictions,
        columns=list(range(len(proba_predictions[0]))),
        index=y_true.index,
    )
    trues = y_true.groupby(level=0).agg(lambda x: x.value_counts().index[0])
    predictions = proba_predictions.groupby(level=0).sum().idxmax(axis="columns")
    return balanced_accuracy_score(trues, predictions)


def file_scores(
    y_true: pd.Series,
    predictions: Union[pd.Series, np.array, list],
    sample_weight=None,
) -> float:

    matches = pd.DataFrame(
        data={"true": y_true, "predict": predictions}, index=y_true.index
    )
    matches = matches.groupby(level=0).agg(lambda x: x.value_counts().index[0])
    return balanced_accuracy_score(matches["true"], matches["predict"])


def file_score(
    y_true: Union[pd.Series, np.array, list],
    predictions: Union[pd.Series, np.array, list],
) -> float:
    if len(set(y_true)) != 1:
        raise ValueError(f"Multiple defects detected: {y_true}")
    if isinstance(y_true, pd.Series):
        correct_defect = y_true.iloc[0]
    else:
        correct_defect = y_true[0]
    predictions = pd.Series(data=predictions)
    predicted_defects = predictions.mode().values
    if correct_defect not in predicted_defects:
        return 0.0
    else:
        return 1.0 / len(predicted_defects)


# Adapted from:
# https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_ranking.py#L1572
def top_k_accuracy_score(y_true, predictions, labels, k=constants.K):
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_true = column_or_1d(y_true)
    y_score = check_array(predictions, ensure_2d=False)
    check_consistent_length(y_true, y_score)

    y_score_n_classes = y_score.shape[1] if y_score.ndim == 2 else 2

    labels = column_or_1d(labels)
    classes = _unique(labels)
    n_labels = len(labels)
    n_classes = len(classes)

    if n_classes != n_labels:
        raise ValueError("Parameter 'labels' must be unique.")

    if not np.array_equal(classes, labels):
        raise ValueError("Parameter 'labels' must be ordered.")

    if n_classes != y_score_n_classes:
        raise ValueError(
            f"Number of given labels ({n_classes}) not equal to the "
            f"number of classes in 'y_score' ({y_score_n_classes})."
        )

    if len(np.setdiff1d(y_true, classes)):
        raise ValueError("'y_true' contains labels not in parameter 'labels'.")

    y_true_encoded = _encode(y_true, uniques=classes)
    sorted_pred = np.argsort(y_score, axis=1, kind="mergesort")[:, ::-1]
    hits = (y_true_encoded == sorted_pred[:, :k].T).any(axis=0)

    return np.average(hits)


class MyScorer(_BaseScorer):
    def dummy():  # type: ignore
        raise NotImplementedError("")

    def __init__(self):
        super().__init__(MyScorer.dummy, 1, {})

    # Adapted from: scikit-learn/sklearn/metrics/_scorer.py
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        if no_predict_proba(get_classifier(estimator)):
            return file_scores(y_true, method_caller(estimator, "predict", X))

        y_type = type_of_target(y_true)
        y_pred = method_caller(estimator, "predict_proba", X)
        if y_type == "binary" and y_pred.shape[1] <= 2:
            y_pred = self._select_proba_binary(y_pred, estimator.classes_)
        return avg_file_scores(y_true, y_pred)
