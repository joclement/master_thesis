import click
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)

from . import __version__


TEST_SET_IDS = ["/cleanair/", "/DC-GIL/", "/normal/", "/normal-0.4/"]


def print_scores(test_predictions) -> None:
    true = test_predictions["true"]
    prediction = test_predictions["prediction"]
    click.echo(f"  Balanced accuracy: {balanced_accuracy_score(true, prediction)}")
    click.echo(f"  Accuracy: {accuracy_score(true, prediction)}")
    click.echo(f"  Support: {len(true)}")


def main(test_predictions: pd.DataFrame) -> None:
    for test_set_id in TEST_SET_IDS:
        part = test_predictions.loc[test_predictions.index.str.contains(test_set_id)]
        click.echo(f"Scores for {test_set_id} set:")
        print_scores(part)


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "test_predictions", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def click_command(test_predictions):
    main(pd.read_csv(test_predictions, header=0, index_col="path"))
