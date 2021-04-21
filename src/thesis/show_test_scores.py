from pathlib import Path

import click
import matplotlib.pyplot as plt
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


TEST_SET_IDS = ["/normal/", "/DC-GIL/", "/normal-0.4/", "/cleanair/"]

ID_TO_NAME = {
    "normal": "Mixture",
    "normal-0.4": "Mix-0.4nV",
    "DC-GIL": "OS-V2",
    "cleanair": "CleanAir",
}


def build_df(results: pd.DataFrame, test_set_name: str) -> pd.DataFrame:
    defects = sorted(set(results["true"]))
    args = {
        "average": None,
        "labels": defects,
        "zero_division": 0,
    }
    defect_names = [Defect(d).abbreviation() for d in defects]
    return pd.DataFrame(
        data={
            "Dataset": [test_set_name] * 2 * len(defects),
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


def create_plots(
    results: pd.DataFrame,
    test_set_name: str,
    test_results_name: str,
    output_dir: Path,
    show: bool,
) -> None:
    sns.set(font_scale=1.5)

    ax = sns.catplot(
        data=build_df(results, test_set_name),
        x="Defect class",
        y="score",
        hue="metric",
        kind="bar",
    )
    ax.set(ylim=(0, 1), xlabel="")
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


def create_combined_plot(
    test_predictions: pd.DataFrame, test_results_name: str, output_dir: Path, show: bool
):
    scores_df = pd.concat(
        [
            build_df(
                test_predictions.loc[test_predictions.index.str.contains(test_set_id)],
                test_set_id.strip("/"),
            )
            for test_set_id in TEST_SET_IDS
        ]
    )

    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 7), (0, 0), colspan=6, fig=fig)
    ax2 = plt.subplot2grid((2, 7), (0, 6), colspan=1, fig=fig)
    ax3 = plt.subplot2grid((2, 7), (1, 0), colspan=5, fig=fig, sharey=ax1)
    ax4 = plt.subplot2grid((2, 7), (1, 5), colspan=2, fig=fig, sharey=ax2)
    axes = [ax1, ax2, ax3, ax4]

    WIDTH = 0.35
    for i, (dataset_name, df) in enumerate(
        scores_df.groupby(axis=0, by="Dataset", sort=False)
    ):
        ax = axes[i]
        x = np.arange(len(df["score"]) / 2)
        metrics = list(df.groupby(axis=0, by="metric", sort=False))
        ax.bar(x - WIDTH / 2, metrics[0][1]["score"], WIDTH, label=metrics[0][0])
        ax.bar(x + WIDTH / 2, metrics[1][1]["score"], WIDTH, label=metrics[1][0])
        if ax.is_first_col():
            ax.set_ylabel("Score")
        else:
            ax.set_yticklabels([])
        ax.set_title(ID_TO_NAME[dataset_name])
        ax.set_xticks(x)
        ax.set_ylim([0, 1])
        ax.set_xticklabels(df["Defect class"].unique())

    ax1.legend(loc="lower left")
    fig.tight_layout()

    util.finish_plot(f"{test_results_name}_combined_recall_precision", output_dir, show)


def main(test_predictions: pd.DataFrame, test_results_name, show: bool = False) -> None:

    output_dir = Path(f"./output/test_plots/{test_results_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    create_combined_plot(test_predictions, test_results_name, output_dir, show)

    for test_set_id in TEST_SET_IDS:
        part = test_predictions.loc[test_predictions.index.str.contains(test_set_id)]
        click.echo(f"Scores for {test_set_id} set:")
        print_scores(part)
        create_plots(part, test_set_id.strip("/"), test_results_name, output_dir, show)


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "test_predictions", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--show/--no-show", default=False)
def click_command(test_predictions, show):
    main(
        pd.read_csv(test_predictions, header=0, index_col="path"),
        Path(test_predictions).stem,
        show,
    )
