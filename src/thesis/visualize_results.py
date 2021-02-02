from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from . import __version__, util
from .constants import (
    CONFIG_FILENAME,
    CONFIG_MODELS_RUN_ID,
    SCORES_FILENAME,
)


def make_pandas_plot(scores: pd.DataFrame, config: dict, description: str):
    if config["general"]["cv"] == "logo":
        xerr = None
    else:
        xerr = scores.std(axis=1, level=0)
    scores.mean(axis=1, level=0).plot.barh(title=description, xerr=xerr)


def plot_results(
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
        classify_config = yaml.safe_load(stream)
    if config_file:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
        models = config["models"]
    else:
        models = classify_config[CONFIG_MODELS_RUN_ID]

    scores = pd.read_csv(Path(result_dir, SCORES_FILENAME), header=[0, 1], index_col=0)
    scores = scores.loc[scores.index.isin(models), :]
    plot_results(scores, classify_config, show=show)
