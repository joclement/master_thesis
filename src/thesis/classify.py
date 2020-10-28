import click
import pandas as pd
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsClassifier

from . import __version__, classifiers, data, fingerprint

FINGERPRINTS = [fingerprint.lukas, fingerprint.tu_graz]

CLASSIFIERS = [
    KNeighborsClassifier(n_neighbors=1),
    classifiers.LukasMeanDist(),
    svm.SVC(decision_function_shape="ovo"),
]


@click.command()
@click.version_option(version=__version__)
@click.argument("directory", type=click.Path(exists=True))
def main(directory):
    """Print measurement info on given measurement file or folder

    PATH file or folder to read csv files for classification from
    """

    measurements, _ = data.read_recursive(directory)
    data.clip_neg_pd_values(measurements)
    train, test = data.split_train_test(measurements)

    normalizer = data.Normalizer(train)
    normalizer.apply(train)
    normalizer.apply(test)

    defect_names = [data.Defect(d).name for d in sorted(set(data.get_defects(test)))]
    accuracies = pd.DataFrame(
        {f.__name__: list(range(len(CLASSIFIERS))) for f in FINGERPRINTS},
        index=[type(c).__name__ for c in CLASSIFIERS],
    )
    for finger_algo in FINGERPRINTS:
        train_fingers = fingerprint.build_set(train, finger_algo)
        test_fingers = fingerprint.build_set(test, finger_algo)

        x_train = train_fingers.drop(data.CLASS, axis=1)
        y_train = train_fingers[data.CLASS]
        x_test = test_fingers.drop(data.CLASS, axis=1)
        y_test = test_fingers[data.CLASS]

        for classifier in CLASSIFIERS:
            classifier.fit(x_train, y_train)
            predictions = classifier.predict(x_test)

            accuracy = metrics.accuracy_score(y_test, predictions)
            accuracies.loc[type(classifier).__name__, finger_algo.__name__] = accuracy
            click.echo(
                f"Accuracy for {type(classifier).__name__}"
                f" with fingerprint {finger_algo.__name__}: {accuracy}"
            )

            click.echo(f"Confusion matrix for {type(classifier).__name__}:")
            confusion_matrix = pd.DataFrame(
                metrics.confusion_matrix(y_test, predictions),
                index=defect_names,
                columns=defect_names,
            )
            click.echo(confusion_matrix.to_string())

            click.echo()
            click.echo(" ============================================================ ")
            click.echo()

    click.echo(accuracies)
