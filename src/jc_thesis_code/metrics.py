from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.multiclass import type_of_target

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
