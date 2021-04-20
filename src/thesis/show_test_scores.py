from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
)

from . import __version__, util
from .data import Defect


TEST_SET_IDS = ["/cleanair/", "/DC-GIL/", "/normal/", "/normal-0.4/"]


def create_plots(results: pd.DataFrame, test_set_name, test_results_name, show) -> None:
    sns.set(font_scale=1.5)
    defects = sorted(set(results["true"]))
    args = {
        "average": None,
        "labels": defects,
        "zero_division": 0,
    }
    defect_names = [Defect(d).abbreviation() for d in defects]
    df = pd.DataFrame(
        data={
            "metric": util.flatten(
                [["Recall"] * len(defects), ["Precision"] * len(defects)]
            ),
            "score": np.concatenate(
                [
                    recall_score(results["true"], results["prediction"], **args),
                    precision_score(results["true"], results["prediction"], **args),
                ]
            ),
            "Defect class": util.flatten([defect_names, defect_names]),
        }
    )

    ax = sns.barplot(data=df, x="Defect class", y="score", hue="metric")
    ax.set_ylim([0, 1])
    ax.set(xlabel="")
    output_dir = Path(f"./output/test_plots/{test_results_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    util.finish_plot(
        f"{test_results_name}_{test_set_name}_recall_precision", output_dir, show
    )


def print_scores(test_predictions) -> None:
    true = test_predictions["true"]
    prediction = test_predictions["prediction"]
    click.echo(f"  Balanced accuracy: {balanced_accuracy_score(true, prediction)}")
    click.echo(f"  Accuracy: {accuracy_score(true, prediction)}")
    click.echo(f"  Support: {len(true)}")


def main(test_predictions: pd.DataFrame, test_results_name, show: bool = False) -> None:
    for test_set_id in TEST_SET_IDS:
        part = test_predictions.loc[test_predictions.index.str.contains(test_set_id)]
        click.echo(f"Scores for {test_set_id} set:")
        print_scores(part)
        create_plots(part, test_set_id.strip("/"), test_results_name, show)


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "test_predictions", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--show/--no-show", default=True)
def click_command(test_predictions, show):
    main(
        pd.read_csv(test_predictions, header=0, index_col="path"),
        Path(test_predictions).stem,
        show,
    )
