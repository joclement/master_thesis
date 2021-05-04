from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    precision_score,
    recall_score,
)

from . import __version__, util
from .data import Defect


ID_TO_NAME = {
    "normal": "Mixture",
    "normal-0.4": "Mix-0.4nV",
    "DC-GIL": "OS-V2",
    "cleanair": "CleanAir",
    "noise": "Noise",
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


def add_legend(fig: plt.Figure) -> None:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(), loc="center right", prop={"size": 10}
    )


def create_combined_plot(
    test_predictions: pd.DataFrame, test_set_ids: List[str], with_noise: bool
):
    scores_df = pd.concat(
        [
            build_df(
                test_predictions.loc[test_predictions.index.str.contains(test_set_id)],
                test_set_id.strip("/"),
            )
            for test_set_id in test_set_ids
        ]
    )

    fig = plt.figure()
    if with_noise:
        ax1 = plt.subplot2grid((3, 6), (0, 0), colspan=6, fig=fig)
        ax3 = plt.subplot2grid((3, 6), (1, 0), colspan=5, fig=fig, sharey=ax1)
        ax2 = plt.subplot2grid((3, 6), (2, 0), colspan=1, fig=fig, sharey=ax1)
        ax4 = plt.subplot2grid((3, 6), (2, 2), colspan=2, fig=fig)
        ax5 = plt.subplot2grid((3, 6), (2, 5), colspan=1, fig=fig, sharey=ax4)
        axes = [ax1, ax2, ax3, ax4, ax5]
    else:
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
        ax.bar(x - WIDTH / 2, metrics[0][1]["score"], WIDTH - 0.02, label=metrics[0][0])
        ax.bar(x + WIDTH / 2, metrics[1][1]["score"], WIDTH - 0.02, label=metrics[1][0])
        if ax.is_first_col():
            ax.set_ylabel("Score")
        else:
            ax.set_yticklabels([])
        ax.set_title(ID_TO_NAME[dataset_name])
        ax.set_xticks(x)
        ax.set_ylim([0, 1])
        ax.set_xticklabels(df["Defect class"].unique())

    if with_noise:
        add_legend(fig)
    else:
        ax1.legend(loc="lower left")
    fig.tight_layout()


def create_confusion_matrix(results: pd.DataFrame):
    defect_names = [Defect(d).abbreviation() for d in sorted(set(results["true"]))]
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(
        confusion_matrix(results["true"], results["prediction"]),
        display_labels=defect_names,
    ).plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)


def main(
    test_predictions: pd.DataFrame,
    test_results_name,
    with_noise: bool,
    show: bool = False,
) -> None:
    output_dir = Path(f"./output/test_plots/{test_results_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_set_ids = ["/normal/", "/DC-GIL/", "/normal-0.4/", "/cleanair/"]
    if with_noise:
        test_set_ids.append("/noise/")

    create_combined_plot(test_predictions, test_set_ids, with_noise)
    util.finish_plot(f"{test_results_name}_combined_recall_precision", output_dir, show)

    create_confusion_matrix(test_predictions)
    util.finish_plot(f"confusion_matrix_{test_results_name}", output_dir, show)

    for test_set_id in test_set_ids:
        part = test_predictions.loc[test_predictions.index.str.contains(test_set_id)]
        if "duration" in part.columns:
            click.echo(f"Duration sum for {test_set_id} set: {part['duration'].sum()}")
        else:
            click.echo(
                f"Preprocess duration sum for {test_set_id} set: "
                f"{part['preprocess_duration'].sum()}"
            )
            click.echo(
                f"Predict duration sum for {test_set_id} set: "
                f"{part['predict_duration'].sum()}"
            )
        click.echo(f"Scores for {test_set_id} set:")
        print_scores(part)
        create_plots(part, test_set_id.strip("/"), test_results_name, output_dir, show)


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "test_predictions", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
@click.option("--with-noise/--without-noise", default=False)
@click.option("--show/--no-show", default=False)
def click_command(test_predictions, with_noise, show):

    main(
        pd.read_csv(test_predictions, header=0, index_col="path"),
        Path(test_predictions).stem,
        with_noise,
        show,
    )
