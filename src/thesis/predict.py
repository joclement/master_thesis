from pathlib import Path
import pickle
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import Defect
from .models import is_data_finger


def load_mlp(model_paths: List[Path]) -> Pipeline:
    # TODO implement
    return None


class PredictionHandler:
    def __init__(
        self,
        preprocessor_path: Path,
        model_paths: List[Path],
        finger_preprocessor_path: Optional[Path] = None,
    ):
        with open(preprocessor_path, "rb") as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)
        if len(model_paths) > 1:
            self.classifier = load_mlp(model_paths)
        else:
            with open(model_paths[0], "rb") as model_file:
                self.classifier = pickle.load(model_file)
        if is_data_finger(self.classifier.steps[0][0]):
            if finger_preprocessor_path is not None:
                with open(finger_preprocessor_path, "rb") as finger_preprocessor_file:
                    self.finger_preprocessor = pickle.load(finger_preprocessor_file)
            else:
                raise ValueError("Missing finger preprocessor.")

    def predict_one(self, csv_data: pd.DataFrame) -> Tuple[Defect, Any]:
        X = self.preprocessor.transform([csv_data])
        if hasattr(self, "finger_preprocessor"):
            X = self.finger_preprocessor.transform(X)
        proba_predictions = self.classifier.predict_proba(X)
        proba_prediction = np.sum(proba_predictions, axis=0) / len(X)
        prediction = np.argmax(proba_prediction)
        return prediction, proba_prediction
