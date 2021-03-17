from pathlib import Path
from typing import Final, Optional
from warnings import warn

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import __version__, data
from .classify import build_index
from .constants import K
from .data import Defect
from .metrics import top_k_accuracy_score
from .predict import PredictionHandler
from .visualize_results import plot_predictions


def print_score(name: str, value: float) -> None:
    click.echo(f"{name}: {value:.3f}")


def main(
    test_folder: Path,
    preprocessor_file: Path,
    model_file: Path,
    finger_preprocessor_file: Optional[Path],
    show: bool = False,
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
    defects = list(Defect)
    defects.remove(Defect.noise)
    print_score(
        f"Top {K} accuracy",
        top_k_accuracy_score(y, proba_predictions, k=K, labels=defects),
    )
    predictions = pd.DataFrame(
        data={"model": predictions},
        index=build_index(measurements),
    )
    plot_predictions(predictions, show=show)


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
@click.option("--show/--no-show", default=False)
def click_command(
    test_folder,
    preprocessor,
    model,
    finger_preprocessor,
    show,
):
    main(test_folder, preprocessor, model, finger_preprocessor, show)
