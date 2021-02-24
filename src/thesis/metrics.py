from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


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
