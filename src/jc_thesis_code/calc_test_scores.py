from pathlib import Path
from typing import List, Optional
from warnings import resetwarnings, warn

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from . import __version__, data
from .classify import build_index
from .predict import PredictionHandler
from .visualize_results import plot_predictions


resetwarnings()


def print_score(name: str, value: float) -> None:
    click.echo(f"{name}: {value:.3f}")


def main(
    test_folders: List[Path],
    preprocessor_file: Path,
    model_file: Path,
    finger_preprocessor_file: Optional[Path] = None,
    keras_model: Optional[Path] = None,
    output_file: Optional[Path] = None,
    show: bool = False,
):
    np.set_printoptions(precision=2)

    measurements = []
    for test_folder in test_folders:
        measurements.extend(
            data.read_recursive(test_folder, data.TreatNegValues.absolute)[0]
        )
    y = np.array(data.get_defects(measurements))

    predictionHandler = PredictionHandler(
        preprocessor_file, model_file, finger_preprocessor_file, keras_model
    )
    predictions = []
    failing_indexes = []
    preprocess_durations = []
    predict_durations = []
    for i, df in enumerate(measurements):
        try:
            (
                prediction,
                proba_prediction,
                preprocess_duration,
                predict_duration,
            ) = predictionHandler.predict_one(df)
            click.echo(
                f"prediction: {prediction}, probas: {proba_prediction}, true: {y[i]}"
            )
        except ValueError as e:
            failing_indexes.append(i)
            warn(str(e) + f" Path: {df.attrs[data.PATH]}")
            continue
        predictions.append(prediction)
        preprocess_durations.append(preprocess_duration)
        predict_durations.append(predict_duration)
    y = np.delete(y, failing_indexes)
    measurements = [df for i, df in enumerate(measurements) if i not in failing_indexes]
    print_score("Accuracy", accuracy_score(y, predictions))
    print_score("Balanced accuracy", balanced_accuracy_score(y, predictions))
    if output_file:
        pd.DataFrame(
            data={
                "prediction": predictions,
                "true": y,
                "preprocess_duration": preprocess_durations,
                "predict_duration": predict_durations,
            },
            index=build_index(measurements),
        ).to_csv(output_file)
    predictions_df = pd.DataFrame(
        data={"predictions": predictions, "true": y},
        index=build_index(measurements),
    )
    plot_predictions(predictions_df, ["predictions"], show=show)


@click.command()
@click.version_option(version=__version__)
@click.option(
    "-t",
    "--test",
    "test_folders",
    type=click.Path(exists=True, file_okay=True),
    help="Test location",
    required=True,
    multiple=True,
)
@click.option(
    "-m",
    "--model",
    type=click.Path(exists=True, file_okay=True),
    help="Pickled model path",
    required=True,
)
@click.option(
    "-p",
    "--preprocessor",
    type=click.Path(exists=True, file_okay=True),
    help="Pickled preprocessor path",
    required=True,
)
@click.option(
    "-f",
    "--finger-preprocessor",
    type=click.Path(exists=True, file_okay=True),
    help="Pickled finger preprocessor path",
)
@click.option(
    "-k",
    "--keras-model",
    type=click.Path(exists=True, file_okay=True),
    help="Keras model saved in H5",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    help="Save predictions",
)
@click.option("--show/--no-show", default=False)
def click_command(
    test_folders,
    preprocessor,
    model,
    keras_model,
    finger_preprocessor,
    output_file,
    show,
):
    main(
        test_folders,
        preprocessor,
        model,
        finger_preprocessor,
        keras_model,
        output_file,
        show,
    )
