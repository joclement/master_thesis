from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics, neural_network, svm
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from . import __version__, classifiers, data, fingerprint

FINGERPRINTS = {
    "Lukas": fingerprint.lukas,
    "TU Graz": fingerprint.tu_graz,
    "Lukas + TU Graz": fingerprint.lukas_plus_tu_graz,
}

CLASSIFIERS = {
    "1-NN": KNeighborsClassifier(n_neighbors=1),
    "3-NN": KNeighborsClassifier(n_neighbors=3),
    "LukasMeanDist": classifiers.LukasMeanDist(),
    "SVM": svm.SVC(decision_function_shape="ovo"),
    "MLP": neural_network.MLPClassifier(
        hidden_layer_sizes=(9,),
        solver="lbfgs",
    ),
}

CV = 4


def _drop_unneded_columns(measurements):
    for measurement in measurements:
        measurement.drop(columns=data.TEST_VOLTAGE, inplace=True, errors="ignore")
        measurement.drop(columns=[data.TIME, data.VOLTAGE_SIGN], inplace=True)


@click.command()
@click.version_option(version=__version__)
@click.argument("input_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path(exists=True))
def main(input_directory, output_directory):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    OUTPUT_DIRECTORY folder where plot(s) will be saved
    """

    measurements, _ = data.read_recursive(input_directory)
    _drop_unneded_columns(measurements)
    data.clip_neg_pd_values(measurements)

    defect_names = [
        data.Defect(d).name for d in sorted(set(data.get_defects(measurements)))
    ]
    accuracies = pd.DataFrame(
        {f: list(range(len(CLASSIFIERS))) for f in FINGERPRINTS.keys()},
        index=[c for c in CLASSIFIERS.keys()],
    )
    for finger_algo_name, finger_algo in FINGERPRINTS.items():
        fingerprints = fingerprint.build_set(measurements, finger_algo)

        X = fingerprints.drop(data.CLASS, axis=1)
        y = fingerprints[data.CLASS]

        for classifier_name, classifier in CLASSIFIERS.items():
            pipe = make_pipeline(MinMaxScaler(), classifier)
            scores = cross_val_score(
                pipe, X, y, cv=CV, scoring="accuracy", error_score="raise"
            )

            accuracies.loc[classifier_name, finger_algo_name] = scores.mean()
            click.echo(
                f"Accuracies for {classifier_name}"
                f" with fingerprint {finger_algo_name}: {scores}"
            )

            predictions = cross_val_predict(pipe, X, y, cv=CV)
            click.echo(f"Confusion matrix for {classifier_name}:")
            confusion_matrix = pd.DataFrame(
                metrics.confusion_matrix(y, predictions),
                index=defect_names,
                columns=defect_names,
            )
            click.echo(confusion_matrix.to_string())

            click.echo()
            click.echo(" ============================================================ ")
            click.echo()

    click.echo(accuracies)

    ax = accuracies.plot.bar(rot=0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by classifier and fingerprint")
    plt.savefig(Path(output_directory, "bar.svg"))
