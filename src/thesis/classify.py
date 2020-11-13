from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, neural_network, svm
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from . import __version__, classifiers, data, fingerprint, util

FINGERPRINTS = {
    "Ott": fingerprint.lukas,
    "TU Graz": fingerprint.tu_graz,
    "Ott + TU Graz": fingerprint.lukas_plus_tu_graz,
}
TS = "Time Series"
DATASET_NAMES = list(FINGERPRINTS.keys()) + [TS]

FINGERPRINT_CLASSIFIERS = {
    "1-NN": KNeighborsClassifier(n_neighbors=1),
    "3-NN": KNeighborsClassifier(n_neighbors=3),
    "Ott": classifiers.LukasMeanDist(),
    "SVM": svm.SVC(decision_function_shape="ovo"),
    "MLP": neural_network.MLPClassifier(
        hidden_layer_sizes=(9,),
        solver="lbfgs",
    ),
}

SEQUENCE_CLASSIFIERS = {
    "1-NN DTW": KNeighborsTimeSeriesClassifier(n_neighbors=1),
    "3-NN DTW": KNeighborsTimeSeriesClassifier(n_neighbors=3),
}

CLASSIFIERS = {**FINGERPRINT_CLASSIFIERS, **SEQUENCE_CLASSIFIERS}

CV = 4
SCORE_METRIC = "balanced_accuracy"
SCORE_METRIC_NAME = SCORE_METRIC.replace("_", " ")


def _echo_visual_break():
    click.echo()
    click.echo(" ============================================================ ")
    click.echo()


def _drop_unneded_columns(measurements):
    for measurement in measurements:
        measurement.drop(columns=data.TEST_VOLTAGE, inplace=True, errors="ignore")
        measurement.drop(columns=[data.TIME, data.VOLTAGE_SIGN], inplace=True)


def _report_confusion_matrix(
    classifier_name: str,
    variation_description: str,
    confusion_matrix,
    defect_names: List[str],
    output_directory: Path,
):
    click.echo(f"Confusion matrix for {classifier_name}:")
    click.echo(
        pd.DataFrame(
            confusion_matrix,
            index=defect_names,
            columns=defect_names,
        ).to_string()
    )

    metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=defect_names).plot()
    filepath = f"confusion_matrix_{classifier_name}_{variation_description}.svg"
    filepath = filepath.replace(" ", "_")
    plt.savefig(
        Path(
            output_directory,
            filepath,
        )
    )


@click.command()
@click.version_option(version=__version__)
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path(exists=True))
@click.option(
    "--calc-cm",
    "-c",
    is_flag=True,
    help="Calculate confusion matrix",
)
def main(input_directory, output_directory, calc_cm: bool):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    OUTPUT_DIRECTORY folder where plot(s) will be saved
    """
    measurements, _ = data.read_recursive(input_directory)
    _drop_unneded_columns(measurements)
    data.clip_neg_pd_values(measurements)

    defect_names = [
        data.DEFECT_NAMES[data.Defect(d)]
        for d in sorted(set(data.get_defects(measurements)))
    ]
    mean_accuracies = pd.DataFrame(
        {f: np.zeros(len(CLASSIFIERS)) for f in DATASET_NAMES},
        index=[c for c in CLASSIFIERS.keys()],
    )
    std_accuracies = mean_accuracies.copy(deep=True)

    min_len_measurements = min([len(m) for m in measurements])
    X = to_time_series_dataset(
        [
            df.drop(data.CLASS, axis=1)[1 : min_len_measurements - 1]
            for df in measurements
        ]
    )
    y = data.get_defects(measurements)

    for classifier_name, classifier in SEQUENCE_CLASSIFIERS.items():
        pipe = make_pipeline(classifier)
        scores = cross_val_score(
            pipe, X, y, cv=CV, scoring=SCORE_METRIC, error_score="raise", n_jobs=-1
        )
        click.echo(f"Scores for {classifier_name} with {TS}: {scores}")

        mean_accuracies.loc[classifier_name, TS] = scores.mean()
        std_accuracies.loc[classifier_name, TS] = scores.std()

        if calc_cm:
            confusion_matrix = metrics.confusion_matrix(
                y, cross_val_predict(pipe, X, y, cv=CV, n_jobs=-1)
            )
            _report_confusion_matrix(
                classifier_name,
                TS,
                confusion_matrix,
                defect_names,
                output_directory,
            )

        _echo_visual_break()

    for finger_algo_name, finger_algo in FINGERPRINTS.items():
        fingerprints = fingerprint.build_set(measurements, finger_algo)

        X = fingerprints.drop(data.CLASS, axis=1)
        y = fingerprints[data.CLASS]

        for classifier_name, classifier in FINGERPRINT_CLASSIFIERS.items():
            pipe = make_pipeline(MinMaxScaler(), classifier)
            scores = cross_val_score(
                pipe, X, y, cv=CV, scoring=SCORE_METRIC, error_score="raise", n_jobs=-1
            )
            click.echo(
                f"Scores for {classifier_name} with fingerprint {finger_algo_name}: "
                f"{scores}"
            )

            mean_accuracies.loc[classifier_name, finger_algo_name] = scores.mean()
            std_accuracies.loc[classifier_name, finger_algo_name] = scores.std()

            if calc_cm:
                predictions = cross_val_predict(pipe, X, y, cv=CV, n_jobs=-1)
                confusion_matrix = metrics.confusion_matrix(y, predictions)
                _report_confusion_matrix(
                    classifier_name,
                    f"fingerprint {finger_algo_name}",
                    confusion_matrix,
                    defect_names,
                    output_directory,
                )

            _echo_visual_break()

    click.echo(mean_accuracies)

    ax = mean_accuracies.plot.bar(rot=0, yerr=std_accuracies)
    ax.set_ylabel(SCORE_METRIC_NAME)
    ax.set_title(
        f"{SCORE_METRIC_NAME} by classifier and fingerprint"
        f" for {CV}-fold CV on {len(measurements)} files"
    )
    ax.legend(loc=3)
    util.finish_plot(f"classifiers_{SCORE_METRIC}_bar", output_directory, False)
