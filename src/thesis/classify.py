import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from . import __version__, data, fingerprint, util


@click.command()
@click.version_option(version=__version__)
@click.argument("directory", type=click.Path(exists=True))
def main(directory):
    """Print measurement info on given measurement file or folder

    PATH file or folder to print measurement info for
    """

    measurements, _ = data.read_recursive(directory)
    train, test = data.split_train_test(measurements)

    normalizer = data.Normalizer(train)
    normalizer.apply(train)
    normalizer.apply(test)

    train_fingers = fingerprint.build_set(train)
    test_fingers = fingerprint.build_set(test)

    x_train = train_fingers.drop(data.CLASS, axis=1)
    y_train = train_fingers[data.CLASS]
    x_test = test_fingers.drop(data.CLASS, axis=1)
    y_test = test_fingers[data.CLASS]

    k_nn = KNeighborsClassifier(n_neighbors=1)
    k_nn.fit(x_train, y_train)
    predictions = k_nn.predict(x_test)

    click.echo(
        f"Accuracy: {metrics.accuracy_score(y_test, predictions)}"
    )

    predictions = [data.Defect(i).name for i in predictions]
    y_test = [data.Defect(i).name for i in y_test]
    defect_labels = [defect.name for defect in data.Defect]

    click.echo(util.print_confusion_matrix(
        metrics.confusion_matrix(y_test, predictions),
        defect_labels))
