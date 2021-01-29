from typing import Union

import numpy as np
import pandas as pd


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
    predicted_defects = predictions.mode()
    if len(predicted_defects) == 1:
        if predicted_defects[0] == correct_defect:
            return 1.0
        else:
            return 0.0
    elif correct_defect in predicted_defects:
        return 1.0 / len(predicted_defects)
    else:
        return 0.0
