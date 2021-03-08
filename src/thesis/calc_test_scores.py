from pathlib import Path
from typing import Final, Optional
from warnings import warn

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import __version__, data
from .constants import K
from .metrics import top_k_accuracy_score
from .predict import PredictionHandler


def print_score(name: str, value: float) -> None:
    click.echo(f"{name}: {value:.2f}")


def main(
    test_folder: Path,
    preprocessor_file: Path,
    model_file: Path,
    finger_preprocessor_file: Optional[Path],
):
    np.set_printoptions(precision=2)

    measurements, _ = data.read_recursive(test_folder, data.TreatNegValues.absolute)
    y: Final = pd.Series(data.get_defects(measurements))

    predictionHandler = PredictionHandler(
        preprocessor_file, [model_file], finger_preprocessor_file
    )
    predictions = []
    proba_predictions = []
    for i, df in enumerate(measurements):
        try:
            prediction, proba_prediction = predictionHandler.predict_one(df)
        except ValueError as e:
            warn(str(e) + f" Path: {df.attrs[data.PATH]}")
            continue
        click.echo(
            f"prediction: {prediction}, probas: {proba_prediction}, true: {y[i]}"
        )
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
@click.argument("test-folder", type=click.Path(exists=True, dir_okay=True))
@click.argument("preprocessor", type=click.Path(exists=True, file_okay=True))
@click.argument("model", type=click.Path(exists=True, file_okay=True))
@click.option(
    "-f",
    "--finger-preprocessor",
    type=click.Path(exists=True, file_okay=True),
    help="Pickled finger preprocessor path",
)
def click_command(
    test_folder,
    preprocessor,
    model,
    finger_preprocessor,
):
    main(test_folder, preprocessor, model, finger_preprocessor)
