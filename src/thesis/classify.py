import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from . import __version__, data, fingerprint


@click.command()
@click.version_option(version=__version__)
@click.argument("directory", type=click.Path(exists=True))
def main(directory):
    """Print measurement info on given measurement file or folder

    PATH file or folder to print measurement info for
    """

    measurements, _ = data.read_recursive(directory)
    finger_set = fingerprint.build_set(measurements)

    x_train, x_test, y_train, y_test = train_test_split(
        finger_set.drop(data.CLASS, axis=1),
        finger_set[data.CLASS],
        test_size=0.3,
        stratify=[int(defect) for defect in finger_set[data.CLASS]],
    )

    k_nn = KNeighborsClassifier(n_neighbors=1)
    k_nn.fit(x_train, y_train)
    score = k_nn.score(x_test, y_test)
    click.echo(score)
