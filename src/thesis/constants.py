from typing import Final

SCORES_FILENAME: Final = "models_scores.csv"
CONFIG_FILENAME: Final = "config.yml"
CONFIG_MODELS_RUN_ID: Final = "models-to-run"

TRAIN_ACCURACY: Final = "train_accuracy"
TRAIN_SCORE: Final = "train"
VAL_SCORE: Final = "val"
ACCURACY_SCORE: Final = "accuracy"
K: Final = 3
TOP_K_ACCURACY: Final = f"top_{K}_accuracy"
FILE_SCORE: Final = "file_score"
METRIC_NAMES: Final = {ACCURACY_SCORE, TOP_K_ACCURACY, TRAIN_ACCURACY}
