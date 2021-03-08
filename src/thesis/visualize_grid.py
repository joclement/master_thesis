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

    for i, rank in enumerate(grid_results["rank_test_score"]):
        if rank <= 10:
            print(f"Rank {rank}:")
            print(f"{grid_results['params'][i]}")
    _, ax = plt.subplots()
    y_pos = np.arange(len(grid_results["mean_test_score"]))
    ax.barh(
        y_pos,
        grid_results["mean_test_score"],
        tick_label=grid_results["rank_test_score"],
        xerr=grid_results["std_test_score"],
        align="center",
    )
    ax.set_yticks(y_pos)
    ax.set_xlabel("file-based Balanced accuracy")
    ax.set_xlabel("file-based Balanced accuracy")
    ax.set_xlim([0, 1])
    util.finish_plot("mean-grid-score", None, True)

    scores = combine_split_scores(grid_results)
    scores = drop_unneeded_classifiers(scores)
    sns.swarmplot(data=scores, x="score", y="combination", hue="run")
    util.finish_plot("mean-grid-score", None, True)

    skip = "classifier__min_child_samples"
    scores = combine_split_scores_relplot(grid_results, skip)
    print(scores)
    sns.relplot(
        data=scores,
        x=skip,
        y="score",
        col="combination",
        col_wrap=4,
        kind="line",
    )
    util.finish_plot("relplot-score", None, True)


def drop_unneeded_classifiers(scores):
    # scores = scores.loc[scores["classifier"] != "mlp", :]
    # scores = scores.loc[scores["classifier"] != "RandomForestClassifier()", :]
    # scores = scores.loc[scores["classifier"] != "KNeighborsClassifier()", :]
    # scores = scores.loc[scores["classifier"] != "SVC()", :]
    return scores


def get_classifier_name(params):
    if "classifier" not in params:
        return "mlp"
    return str(params["classifier"])


def get_params_description(params, skip):
    return ",".join(str(v) for k, v in params.items() if k != skip)


def build_row(grid_results, i, split_idx):
    row = {
        "score": grid_results[f"split{split_idx}_test_score"][i],
        "combination": get_params_description(grid_results["params"][i], ""),
        "run": split_idx,
        "i": i,
    }
    return row


def combine_split_scores(grid_results):
    return pd.DataFrame(
        [
            build_row(grid_results, i, split_idx)
            for i in range(len(grid_results["mean_test_score"]))
            for split_idx in range(4)
            if grid_results["rank_test_score"][i] <= 10
        ]
    )


def build_row_relplot(grid_results, i, skip):
    row = {
        "score": grid_results["mean_test_score"][i],
        "combination": get_params_description(grid_results["params"][i], skip),
        "i": i,
    }
    row.update(grid_results["params"][i])
    return row


def combine_split_scores_relplot(grid_results, skip):
    return pd.DataFrame(
        [
            build_row_relplot(grid_results, i, skip)
            for i in range(len(grid_results["mean_test_score"]))
        ]
    )
