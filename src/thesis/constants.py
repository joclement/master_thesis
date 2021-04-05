from enum import Enum
from typing import Final


PREDICTIONS_FILENAME: Final = "models_predictions.csv"
SCORES_FILENAME: Final = "models_scores.csv"
CONFIG_FILENAME: Final = "config.yml"
CONFIG_MODELS_RUN_ID: Final = "models-to-run"


class DataPart(str, Enum):
    train = "train"
    val = "val"
    test = "test"


MIN_TIME_DIFF = "50us"

ACCURACY_SCORE: Final = "accuracy"
BALANCED_ACCURACY_SCORE: Final = "balanced_accuracy"
K: Final = 3
TOP_K_ACCURACY_SCORE: Final = f"top_{K}_accuracy"
FILE_SCORE: Final = "file_score"
AVG_FILE_SCORE: Final = "avg_file_score"

MODEL_ID = "model"

CACHE_DIR = "./cache"

PART = "part"

RANDOM_STATE = 11
