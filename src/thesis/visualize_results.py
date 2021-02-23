from pathlib import Path
from typing import Optional, Union

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
)
import yaml

from . import __version__, util
from .constants import (
    CONFIG_FILENAME,
    CONFIG_MODELS_RUN_ID,
    PREDICTIONS_FILENAME,
    SCORES_FILENAME,
)
from .data import Defect, DEFECT_NAMES, get_defect


def make_pandas_plot(scores: pd.DataFrame, config: dict, description: str):
    if config["general"]["cv"] == "logo":
        xerr = None
    else:
        xerr = scores.std(axis=1, level=0)
    scores.mean(axis=1, level=0).plot.barh(title=description, xerr=xerr)


def get_defect_from_index(index_entry: Union[tuple, str]):
    if isinstance(index_entry, tuple):
        return get_defect(index_entry[0])
    return get_defect(index_entry)


def print_wrong_files(predictions: pd.DataFrame):
    models = predictions.columns
    predictions["true"] = [get_defect(filename) for filename, _ in predictions.index]

    wrongs = np.zeros(len(predictions.index), dtype=int)
    for model in models:
        wrongs += predictions[model] != predictions["true"]
    predictions["wrongs"] = wrongs
    most_wrongs = predictions.sort_values("wrongs", ascending=False)
    click.echo("Absolutely most wrongly classified filenames:")
    click.echo(most_wrongs["wrongs"].head(23))
    click.echo(
        predictions.groupby(level=0)["wrongs"]
        .mean()
        .sort_values(ascending=False)
        .head(23)
    )


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_file_predictions(
    predictions: pd.DataFrame,
):
    file_predictions = pd.DataFrame(
        data={
            c: predictions.groupby(level=0)[c].agg(lambda x: x.mode()[0])
            for c in predictions.columns
        },
        columns=predictions.columns,
        index=list(dict.fromkeys(predictions.index.droplevel(1))),
    )
    file_predictions["true"] = [
        get_defect_from_index(i) for i in file_predictions.index
    ]
    return file_predictions


def plot_predictions(
    predictions: pd.DataFrame,
    output_dir: Optional[Path] = None,
    description: str = "",
    show: bool = False,
):
    models = predictions.columns
    predictions["true"] = [get_defect_from_index(i) for i in predictions.index]
    defect_names = [DEFECT_NAMES[Defect(d)] for d in sorted(set(predictions["true"]))]

    args = {
        "average": None,
        "labels": sorted(set(predictions["true"])),
    }
    df = pd.DataFrame(
        data={
            "recall": np.concatenate(
                [
                    recall_score(predictions["true"], predictions[model], **args)
                    for model in models
                ]
            ),
            "precision": np.concatenate(
                [
                    precision_score(predictions["true"], predictions[model], **args)
                    for model in models
                ]
            ),
            "model": flatten(
                [len(set(predictions["true"])) * [model] for model in models]
            ),
            "defect": flatten([defect_names for _ in models]),
        }
    )

    sns.barplot(data=df, x="model", y="precision", hue="defect")
    util.finish_plot("precisions", output_dir, show)

    sns.barplot(data=df, x="model", y="recall", hue="defect")
    util.finish_plot("recalls", output_dir, show)


def plot_scores(
    scores: pd.DataFrame,
    config: dict,
    output_dir: Optional[Path] = None,
    description: str = "",
    show: bool = False,
):
    y_pos = np.arange(len(scores.index))

    for metric, metric_scores in scores.groupby(axis=1, level=0):
        fig, ax = plt.subplots()
        if config["general"]["cv"] == "logo":
            xerr = None
        else:
            xerr = metric_scores.std(axis=1)
        ax.barh(
            y_pos,
            metric_scores.mean(axis=1),
            xerr=xerr,
            align="center",
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(scores.index)
        ax.set_xlabel(metric)
        ax.set_title(description)
        util.finish_plot(metric, output_dir, show)


@click.command()
@click.version_option(version=__version__)
@click.argument("result_dir", type=click.Path(exists=True))
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Configuration for result plots",
)
@click.option("--show/--no-show", default=True)
def main(result_dir, config_file, show):
    with open(Path(result_dir, CONFIG_FILENAME), "r") as stream:
        classify_config = yaml.load(stream)
    if config_file:
        with open(config_file, "r") as stream:
            config = yaml.load(stream)
        models = config["models"]
    else:
        models = classify_config[CONFIG_MODELS_RUN_ID]

    scores = pd.read_csv(Path(result_dir, SCORES_FILENAME), header=[0, 1], index_col=0)
    scores = scores.loc[scores.index.isin(models), :]
    plot_scores(scores, classify_config, show=show)

    predictions = pd.read_csv(
        Path(result_dir, PREDICTIONS_FILENAME), header=0, index_col=[0, 1]
    )
    predictions = predictions.loc[:, models]
    plot_predictions(predictions, show=show)
    print_wrong_files(predictions)
    plot_predictions(
        get_file_predictions(predictions), description="file-based", show=show
    )
