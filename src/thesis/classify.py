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
from sklearn.tree import DecisionTreeClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from . import __version__, classifiers, data, fingerprint, util

FINGERPRINTS = {
    "Ott": fingerprint.lukas,
    "TU Graz": fingerprint.tu_graz,
    "Ott + TU Graz": fingerprint.lukas_plus_tu_graz,
}
TS = "Time Series"
ONED_TS = f"1D {TS}"
TWOD_TS = f"2D {TS}"
DATASET_NAMES = list(FINGERPRINTS.keys()) + [ONED_TS, TWOD_TS]

FINGERPRINT_CLASSIFIERS = {
    "1-NN": KNeighborsClassifier(n_neighbors=1),
    "3-NN": KNeighborsClassifier(n_neighbors=3),
    "Ott": classifiers.LukasMeanDist(),
    "SVM": svm.SVC(decision_function_shape="ovo"),
    "MLP": neural_network.MLPClassifier(
        hidden_layer_sizes=(9,),
        solver="lbfgs",
    ),
    "DT": DecisionTreeClassifier(),
}

SEQUENCE_CLASSIFIERS = {
    "1-NN DTW": KNeighborsTimeSeriesClassifier(n_neighbors=1),
    "3-NN DTW": KNeighborsTimeSeriesClassifier(n_neighbors=3),
}

CLASSIFIERS = {
    **FINGERPRINT_CLASSIFIERS,
    **SEQUENCE_CLASSIFIERS,
}

CV = 4
SCORE_METRIC = "balanced_accuracy"
SCORE_METRIC_NAME = SCORE_METRIC.replace("_", " ")

FREQUENCY = pd.tseries.frequencies.to_offset("1000us")

N_JOBS = -2


def _echo_visual_break():
    click.echo()
    click.echo(" ============================================================ ")
    click.echo()


def _drop_unneded_columns(measurements):
    cleaned_measurements = []
    for measurement in measurements:
        new_measurement = measurement.drop(columns=data.TEST_VOLTAGE, errors="ignore")
        new_measurement.drop(columns=[data.TIME, data.VOLTAGE_SIGN], inplace=True)
        cleaned_measurements.append(new_measurement)
    return cleaned_measurements


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


def _get_defect_names(measurements: List[pd.DataFrame]):
    return [
        data.DEFECT_NAMES[data.Defect(d)]
        for d in sorted(set(data.get_defects(measurements)))
    ]


def _convert_to_time_series(df: pd.DataFrame) -> pd.Series:
    df[data.TIME] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
    df.set_index(data.TIME, inplace=True)
    time_series = df[data.PD]
    return time_series.asfreq(FREQUENCY, fill_value=0.0)


class ClassificationHandler:
    def __init__(
        self, measurements: List[pd.DataFrame], output_directory, calc_cm: bool
    ):
        self.mean_accuracies = pd.DataFrame(
            {c: np.empty(len(DATASET_NAMES)) for c in CLASSIFIERS.keys()},
            index=[c for c in DATASET_NAMES],
        )
        self.std_accuracies = self.mean_accuracies.copy(deep=True)

        self.measurements = measurements
        self.calc_cm = calc_cm
        self.output_directory = output_directory

    def _cross_validate(
        self, classifier_name, pipeline, X, y, variation, variation_description=None
    ):
        if variation_description is None:
            variation_description = variation

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=CV,
            scoring=SCORE_METRIC,
            error_score="raise",
            n_jobs=N_JOBS,
        )
        click.echo(
            f"Scores for {classifier_name} with {variation_description}: {scores}"
        )

        self.mean_accuracies.loc[variation, classifier_name] = scores.mean()
        self.std_accuracies.loc[variation, classifier_name] = scores.std()

        if self.calc_cm:
            confusion_matrix = metrics.confusion_matrix(
                y, cross_val_predict(pipeline, X, y, cv=CV, n_jobs=N_JOBS)
            )
            _report_confusion_matrix(
                classifier_name,
                variation_description,
                confusion_matrix,
                _get_defect_names(self.measurements),
                self.output_directory,
            )

    def do_1d_sequence_classification(self):
        time_serieses = [
            _convert_to_time_series(df.drop(data.CLASS, axis=1)[1:])
            for df in self.measurements
        ]
        min_len = min([len(time_series) for time_series in time_serieses])
        X = to_time_series_dataset(
            np.array([time_series[:min_len] for time_series in time_serieses])
        )
        y = data.get_defects(self.measurements)

        for classifier_name, classifier in SEQUENCE_CLASSIFIERS.items():
            self._cross_validate(
                classifier_name, make_pipeline(classifier), X, y, ONED_TS
            )
            _echo_visual_break()

    def do_2d_sequence_classification(self):
        measurements = _drop_unneded_columns(self.measurements)
        min_len_measurements = min([len(m) for m in measurements])
        X = to_time_series_dataset(
            [df.drop(data.CLASS, axis=1)[1:min_len_measurements] for df in measurements]
        )
        y = data.get_defects(measurements)

        for classifier_name, classifier in SEQUENCE_CLASSIFIERS.items():
            self._cross_validate(
                classifier_name, make_pipeline(classifier), X, y, TWOD_TS
            )
            _echo_visual_break()

    def do_fingerprint_classification(self):
        measurements = _drop_unneded_columns(self.measurements)
        for finger_algo_name, finger_algo in FINGERPRINTS.items():
            fingerprints = fingerprint.build_set(measurements, finger_algo)

            X = fingerprints.drop(data.CLASS, axis=1)
            y = fingerprints[data.CLASS]

            for classifier_name, classifier in FINGERPRINT_CLASSIFIERS.items():
                self._cross_validate(
                    classifier_name,
                    make_pipeline(MinMaxScaler(), classifier),
                    X,
                    y,
                    finger_algo_name,
                    f"fingerprint {finger_algo_name}",
                )
                _echo_visual_break()


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
    data.clip_neg_pd_values(measurements)

    classificationHandler = ClassificationHandler(
        measurements, output_directory, calc_cm
    )
    classificationHandler.do_1d_sequence_classification()

    classificationHandler.do_2d_sequence_classification()

    classificationHandler.do_fingerprint_classification()

    click.echo(classificationHandler.mean_accuracies)

    ax = classificationHandler.mean_accuracies.plot.bar(
        rot=0, yerr=classificationHandler.std_accuracies
    )
    ax.set_ylabel(SCORE_METRIC_NAME)
    ax.set_title(
        f"{SCORE_METRIC_NAME} by classifier and fingerprint"
        f" for {CV}-fold CV on {len(measurements)} files"
    )
    ax.legend(loc=3)
    util.finish_plot(f"classifiers_{SCORE_METRIC}_bar", output_directory, False)
