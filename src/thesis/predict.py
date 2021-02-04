from pathlib import Path
import pickle
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import Defect


def load_mlp(model_paths: List[Path]) -> Pipeline:
    # TODO implement
    return None


class PredictionHandler:
    def __init__(self, preprocessor_path: Path, model_paths: List[Path]):
        with open(preprocessor_path, "rb") as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)
        if len(model_paths) > 1:
            self.classifier = load_mlp(model_paths)
        else:
            with open(model_paths[0], "rb") as model_file:
                self.classifier = pickle.load(model_file)

    def predict_one(self, csv_data: pd.DataFrame) -> Tuple[Defect, Any]:
        X = self.preprocessor.transform([csv_data])
        proba_predictions = self.classifier.predict_proba(X)
        proba_prediction = np.sum(proba_predictions, axis=0) / len(X)
        prediction = np.argmax(proba_prediction)
        return prediction, proba_prediction
