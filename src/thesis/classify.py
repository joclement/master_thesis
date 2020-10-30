import click
import pandas as pd
from sklearn import metrics, neural_network, svm
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from . import __version__, classifiers, data, fingerprint

FINGERPRINTS = [fingerprint.lukas, fingerprint.tu_graz, fingerprint.lukas_with_tu_graz]

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=1),
    classifiers.LukasMeanDist(),
    svm.SVC(decision_function_shape="ovo"),
    neural_network.MLPClassifier(
        hidden_layer_sizes=(9,),
        solver="lbfgs",
    ),
]


def _drop_unneded_columns(measurements):
    for measurement in measurements:
        measurement.drop(columns=data.TEST_VOLTAGE, inplace=True, errors="ignore")
        measurement.drop(columns=[data.TIME, data.VOLTAGE_SIGN], inplace=True)


@click.command()
@click.version_option(version=__version__)
@click.argument("directory", type=click.Path(exists=True))
def main(directory):
    """Print measurement info on given measurement file or folder

    PATH folder containing csv files for classification
    """

    measurements, _ = data.read_recursive(directory)
    _drop_unneded_columns(measurements)
    data.clip_neg_pd_values(measurements)

    defect_names = [
        data.Defect(d).name for d in sorted(set(data.get_defects(measurements)))
    ]
    accuracies = pd.DataFrame(
        {f.__name__: list(range(len(CLASSIFIERS))) for f in FINGERPRINTS},
        index=[type(c).__name__ for c in CLASSIFIERS],
    )
    for finger_algo in FINGERPRINTS:
        fingerprints = fingerprint.build_set(measurements, finger_algo)

        X = fingerprints.drop(data.CLASS, axis=1)
        y = fingerprints[data.CLASS]

        for classifier in CLASSIFIERS:
            pipe = make_pipeline(MinMaxScaler(), classifier)
            scores = cross_val_score(
                pipe, X, y, cv=4, scoring="accuracy", error_score="raise"
            )

            accuracies.loc[
                type(classifier).__name__, finger_algo.__name__
            ] = scores.mean()
            click.echo(
                f"Accuracies for {type(classifier).__name__}"
                f" with fingerprint {finger_algo.__name__}: {scores}"
            )

            predictions = cross_val_predict(pipe, X, y, cv=3)
            click.echo(f"Confusion matrix for {type(classifier).__name__}:")
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
