from pathlib import Path
import pickle
import time
from typing import Any, Final, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .data import Defect
from .models import is_data_finger


MODEL_FILES_DIR: Final = Path(Path(__file__).parent, "./model_files")
FINGER_PREPROCESSOR_PATH: Final = Path(MODEL_FILES_DIR, "./finger_preprocessor.p")
PREPROCESSOR_PATH: Final = Path(MODEL_FILES_DIR, "./preprocessor.p")
MODEL_PATH: Final = Path(MODEL_FILES_DIR, "./model.p")


class PredictionHandler:
    def __init__(
        self,
        preprocessor_path: Path,
        model_pipeline_path: Path,
        finger_preprocessor_path: Optional[Path] = None,
        keras_model_path: Optional[Path] = None,
    ):
        with open(preprocessor_path, "rb") as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)
        with open(model_pipeline_path, "rb") as model_file:
            self.classifier = pickle.load(model_file)
        if keras_model_path is not None:
            self.classifier.named_steps["classifier"].model = load_model(
                keras_model_path
            )
        if is_data_finger(self.classifier.steps[0][0]):
            if finger_preprocessor_path is not None:
                with open(finger_preprocessor_path, "rb") as finger_preprocessor_file:
                    self.finger_preprocessor = pickle.load(finger_preprocessor_file)
            else:
                raise ValueError("Missing finger preprocessor.")

    def predict_one(self, csv_data: pd.DataFrame) -> Tuple[Defect, Any, float, float]:
        start = time.process_time()
        X = self.preprocessor.transform([csv_data])
        preprocess_duration = time.process_time() - start
        start = time.process_time()
        if hasattr(self, "finger_preprocessor"):
            X = self.finger_preprocessor.transform(X)
        proba_predictions = self.classifier.predict_proba(X)
        predict_duration = time.process_time() - start
        proba_prediction = np.sum(proba_predictions, axis=0) / len(X)
        prediction = np.argmax(proba_prediction)
        return (
            Defect(prediction),
            proba_prediction,
            preprocess_duration,
            predict_duration,
        )


def load_handler_with_pkg_model():
    return PredictionHandler(
        PREPROCESSOR_PATH,
        MODEL_PATH,
        finger_preprocessor_path=FINGER_PREPROCESSOR_PATH,
    )
