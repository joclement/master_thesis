# Based on:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
#
# Therefore, the license of this file is the license from scikit-learn.
# License: BSD 3-Clause License

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
)

from . import util


cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm
n_splits = 4

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
SMALL_SIZE = 15
MEDIUM_SIZE = 20
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)


def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[0, 100],
    )


def main():
    X = np.random.randn(100, 10)
    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack(
        [[ii] * int(100 * perc) for ii, perc in enumerate(percentiles_classes)]
    )

    _, ax = plt.subplots()
    cv = StratifiedKFold(n_splits)
    plot_cv_indices(cv, X, y, ax, n_splits)
    util.finish_plot("StratifiedKFoldIllustration", "./output", True)
