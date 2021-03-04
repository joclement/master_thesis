import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import __version__, util


@click.command()
@click.version_option(version=__version__)
@click.argument("grid_file", type=click.Path(exists=True))
def main(grid_file):
    with open(grid_file, "rb") as f:
        grid_results = pickle.load(f)
    print(grid_results)

    _, ax = plt.subplots()
    y_pos = np.arange(len(grid_results["mean_test_score"]))
    ax.barh(
        y_pos,
        grid_results["mean_test_score"],
        xerr=grid_results["std_test_score"],
        align="center",
    )
    ax.set_yticks(y_pos)
    ax.set_xlabel("file-based Balanced accuracy")
    ax.set_xlim([0, 1])
    util.finish_plot("mean-grid-score", "./", True)

    sns.swarmplot(data=combine_split_scores(grid_results), x="score", y="combination")
    util.finish_plot("mean-grid-score", "./", True)


def combine_split_scores(grid_results):
    return pd.DataFrame(
        [
            {
                "score": grid_results[f"split{split_idx}_test_score"][i],
                "combination": f"c{i}",
            }
            for i in range(len(grid_results["mean_test_score"]))
            for split_idx in range(4)
        ]
    )
