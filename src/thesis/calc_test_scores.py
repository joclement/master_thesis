from pathlib import Path
from typing import Final

import click
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
)

from . import __version__, data
from .constants import K
from .predict import PredictionHandler


def print_score(name: str, value: float) -> None:
    click.echo(f"{name}: {value:.2f}")


def main(test_folder: Path, preprocessor_file: Path, model_file: Path):
    measurements, _ = data.read_recursive(test_folder, data.TreatNegValues.absolute)
    y: Final = pd.Series(data.get_defects(measurements))

    predictionHandler = PredictionHandler(preprocessor_file, [model_file])
    predictions = []
    proba_predictions = []
    for df in measurements:
        prediction, proba_prediction = predictionHandler.predict_one(df)
        predictions.append(prediction)
        proba_predictions.append(proba_prediction)
    print_score("Accuracy", accuracy_score(y, predictions))
    print_score("Balanced accuracy", balanced_accuracy_score(y, predictions))
    print_score(
        f"Top {K} accuracy",
        top_k_accuracy_score(y, proba_predictions, k=K, labels=list(data.Defect)),
    )


@click.command()
@click.version_option(version=__version__)
@click.argument("test_folder", type=click.Path(exists=True, dir_okay=True))
@click.argument("preprocessor_file", type=click.Path(exists=True, file_okay=True))
@click.argument("model_file", type=click.Path(exists=True, file_okay=True))
def click_command(
    test_folder,
    preprocessor_file,
    model_file,
):
    main(test_folder, preprocessor_file, model_file)
