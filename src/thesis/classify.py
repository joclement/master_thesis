from pathlib import Path
from typing import List

import click
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


def _echo_visual_break():
    click.echo()
    click.echo(" ============================================================ ")
    click.echo()


def _drop_unneded_columns(measurements, other_columns: List[str] = []):
    cleaned_measurements = []
    for measurement in measurements:
        new_measurement = measurement.drop(
            columns=[data.TEST_VOLTAGE, *other_columns], errors="ignore"
        )
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
    filestem = f"confusion_matrix_{classifier_name}_{variation_description}"
    filestem = filestem.replace(" ", "_")
    util.finish_plot(filestem, output_directory)


def _get_defect_names(measurements: List[pd.DataFrame]):
    return [
        data.DEFECT_NAMES[data.Defect(d)]
        for d in sorted(set(data.get_defects(measurements)))
    ]


class ClassificationHandler:
    def __init__(
        self, measurements: List[pd.DataFrame], output_directory, calc_cm: bool
    ):
        self.FINGERPRINTS = {
            "Ott": fingerprint.lukas,
            "TU Graz": fingerprint.tu_graz,
            "Ott + TU Graz": fingerprint.lukas_plus_tu_graz,
        }
        TS = "TS"
        self.ONED_TS = f"1D {TS}"
        self.TWOD_TS = f"2D {TS}"
        self.FINGER_SEQUENCES = {
            f"{finger} Seq": algo for finger, algo in self.FINGERPRINTS.items()
        }

        self.FINGERPRINT_CLASSIFIERS = {
            ("1-NN", KNeighborsClassifier(n_neighbors=1)),
            ("3-NN", KNeighborsClassifier(n_neighbors=3)),
            ("Ott Algo", classifiers.LukasMeanDist()),
            ("SVM", svm.SVC(decision_function_shape="ovo")),
            (
                "MLP",
                neural_network.MLPClassifier(hidden_layer_sizes=(9,), solver="lbfgs"),
            ),
            ("DT", DecisionTreeClassifier()),
        }
        self.SEQUENCE_CLASSIFIERS = {
            ("1-NN", KNeighborsTimeSeriesClassifier(n_neighbors=1)),
            ("3-NN", KNeighborsTimeSeriesClassifier(n_neighbors=3)),
        }

        DATASET_NAMES = list(self.FINGERPRINTS.keys()) + [
            self.ONED_TS,
            self.TWOD_TS,
            *self.FINGER_SEQUENCES,
        ]
        CLASSIFIERS = self.FINGERPRINT_CLASSIFIERS | self.SEQUENCE_CLASSIFIERS
        self.mean_accuracies = pd.DataFrame(
            {c[0]: np.empty(len(DATASET_NAMES)) for c in CLASSIFIERS},
            index=[c for c in DATASET_NAMES],
        )
        self.std_accuracies = self.mean_accuracies.copy(deep=True)

        self.CV = 4
        self.SCORE_METRIC = "balanced_accuracy"

        self.MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")
        self.FREQUENCY = pd.tseries.frequencies.to_offset("1s")
        self.FINGERPRINT_SEQUENCE_DURATION = pd.Timedelta("30 seconds")

        self.N_JOBS = -2

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
            cv=self.CV,
            scoring=self.SCORE_METRIC,
            error_score="raise",
            n_jobs=self.N_JOBS,
        )
        click.echo(
            f"Scores for {classifier_name} with {variation_description}: {scores}"
        )

        self.mean_accuracies.loc[variation, classifier_name] = scores.mean()
        self.std_accuracies.loc[variation, classifier_name] = scores.std()

        if self.calc_cm:
            confusion_matrix = metrics.confusion_matrix(
                y, cross_val_predict(pipeline, X, y, cv=self.CV, n_jobs=self.N_JOBS)
            )
            _report_confusion_matrix(
                classifier_name,
                variation_description,
                confusion_matrix,
                _get_defect_names(self.measurements),
                self.output_directory,
            )

    def _convert_to_time_series(self, df: pd.DataFrame) -> pd.Series:
        df[data.TIME] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
        df.set_index(data.TIME, inplace=True)
        time_series = df[data.PD]
        return (
            time_series.asfreq(self.MAX_FREQUENCY, fill_value=0.0)
            .resample(self.FREQUENCY)
            .max()
        )

    def do_1d_sequence_classification(self):
        time_serieses = [
            self._convert_to_time_series(df.drop(data.CLASS, axis=1))
            for df in self.measurements
        ]
        min_len = min([len(time_series) for time_series in time_serieses])
        X = to_time_series_dataset(
            np.array([time_series[:min_len] for time_series in time_serieses])
        )
        y = data.get_defects(self.measurements)

        for classifier_name, classifier in self.SEQUENCE_CLASSIFIERS:
            self._cross_validate(
                classifier_name, make_pipeline(classifier), X, y, self.ONED_TS
            )
            _echo_visual_break()

    def do_2d_sequence_classification(self):
        measurements = _drop_unneded_columns(self.measurements, data.PD_DIFF)
        min_len_measurements = min([len(m) for m in measurements])
        X = to_time_series_dataset(
            [
                df.drop(data.CLASS, axis=1)[1 : 2 * min_len_measurements]
                for df in measurements
            ]
        )
        y = data.get_defects(measurements)

        for classifier_name, classifier in self.SEQUENCE_CLASSIFIERS:
            self._cross_validate(
                classifier_name, make_pipeline(classifier), X, y, self.TWOD_TS
            )
            _echo_visual_break()

    def _build_fingerprint_sequence(self, df: pd.DataFrame, finger_algo):
        timedelta_sum = pd.Timedelta(0)
        index_sequence_splits = []
        for index, value in df[data.TIME_DIFF][1:].iteritems():
            timedelta_sum += pd.Timedelta(value, unit=data.TIME_UNIT)
            if timedelta_sum >= self.FINGERPRINT_SEQUENCE_DURATION:
                index_sequence_splits.append(index)
                timedelta_sum = pd.Timedelta(0)
        sequence = [df.iloc[: index_sequence_splits[0]]]
        for idx in range(1, len(index_sequence_splits)):
            sequence.append(
                df.iloc[index_sequence_splits[idx - 1] : index_sequence_splits[idx]]
            )

        too_short_indexes = []
        for index, sub_df in enumerate(sequence):
            if len(sub_df.index) <= 2:
                too_short_indexes.append(index)

        if len(too_short_indexes) > 0:
            if (
                len(too_short_indexes) - len(range(too_short_indexes[0], len(sequence)))
                <= 1
            ):
                del sequence[too_short_indexes[0] :]
            else:
                raise ValueError(
                    f"Invalid measurement file: {self.FINGERPRINT_SEQUENCE_DURATION}"
                )

        assert all([len(sub_df.index) >= 3 for sub_df in sequence])
        assert all([not sub_df.index.isnull().any() for sub_df in sequence])
        assert len(sequence) >= 3

        return fingerprint.build_set(sequence, finger_algo, False).to_numpy()

    def do_fingerprint_sequence_classification(self):
        measurements = _drop_unneded_columns(self.measurements)
        y = data.get_defects(measurements)

        measurements = [df.drop(data.CLASS, axis=1) for df in measurements]
        for finger_sequence_id, finger_algo in self.FINGER_SEQUENCES.items():
            X = to_time_series_dataset(
                [
                    self._build_fingerprint_sequence(df, finger_algo)
                    for df in measurements
                ]
            )
            assert not np.isinf(X).any()
            assert np.isnan(X).any()

            for classifier_name, classifier in self.SEQUENCE_CLASSIFIERS:
                self._cross_validate(
                    classifier_name, make_pipeline(classifier), X, y, finger_sequence_id
                )
                _echo_visual_break()

    def do_fingerprint_classification(self):
        measurements = _drop_unneded_columns(self.measurements)

        for finger_algo_name, finger_algo in self.FINGERPRINTS.items():
            fingerprints = fingerprint.build_set(measurements, finger_algo)

            X = fingerprints.drop(data.CLASS, axis=1)
            y = fingerprints[data.CLASS]

            for classifier_name, classifier in self.FINGERPRINT_CLASSIFIERS:
                self._cross_validate(
                    classifier_name,
                    make_pipeline(MinMaxScaler(), classifier),
                    X,
                    y,
                    finger_algo_name,
                    f"fingerprint {finger_algo_name}",
                )
                _echo_visual_break()

    def plot_results(self):
        click.echo(self.mean_accuracies)

        ax = self.mean_accuracies.plot.bar(rot=0, yerr=self.std_accuracies)
        score_metric_name = self.SCORE_METRIC.replace("_", " ")
        ax.set_ylabel(score_metric_name)
        ax.set_title(
            f"{score_metric_name} by classifier and fingerprint"
            f" for {self.CV}-fold CV on {len(self.measurements)} files"
            f" for {len(_get_defect_names(self.measurements))} defects"
        )
        ax.legend(loc=3)
        util.finish_plot(
            f"classifiers_{self.SCORE_METRIC}_bar", self.output_directory, False
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
    data.clip_neg_pd_values(measurements)

    classificationHandler = ClassificationHandler(
        measurements, output_directory, calc_cm
    )
    # FIXME Issue #38
    classificationHandler.do_1d_sequence_classification()
    classificationHandler.do_2d_sequence_classification()
    classificationHandler.do_fingerprint_sequence_classification()
    classificationHandler.do_fingerprint_classification()
    classificationHandler.plot_results()
