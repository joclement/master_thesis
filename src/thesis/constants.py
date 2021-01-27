from typing import Final

SCORES_FILENAME: Final = "models_scores.csv"
CONFIG_FILENAME: Final = "config.yml"
CONFIG_MODELS_RUN_ID: Final = "models-to-run"

TRAIN_ACCURACY = "train_accuracy"
TRAIN_SCORE = "train"
VAL_SCORE = "val"
ACCURACY_SCORE = "accuracy"
TOP_K_ACCURACY = "top_3_accuracy"
METRIC_NAMES = {ACCURACY_SCORE, TOP_K_ACCURACY}
