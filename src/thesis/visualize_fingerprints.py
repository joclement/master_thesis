from typing import List

import click
import pandas as pd
from scipy.cluster import hierarchy
import seaborn as sns

from . import __version__, data, fingerprint, util


def _generate_heatmap(fingerprints: pd.DataFrame, output_folder, show):
    corr = fingerprints.corr()
    sns.heatmap(corr, center=0)
    util.finish_plot("correlation_fingerprint_ott+tu_graz", output_folder, show)


def _generate_pairplots(fingerprints: pd.DataFrame, output_folder, show):
    for group in fingerprint.Group:
        parameters = fingerprint.get_parameter_group(fingerprints, group)
        parameters[data.CLASS] = data.get_names(fingerprints[data.CLASS])

        pairgrid = sns.pairplot(parameters, hue=data.CLASS, diag_kind="hist")
        pairgrid.fig.suptitle(
            f"Pairwise relationships in fingerprint parameters related to {group}",
            y=1.08,
        )
        util.finish_plot(f"pairplot_{group}", output_folder, show)


def _generate_dendogram(measurements: List[pd.DataFrame], output_folder, show):
    fingerprints = fingerprint.build_set(measurements, fingerprint.ott, add_class=True)
    Z = hierarchy.linkage(fingerprints.drop(data.CLASS, axis=1), "single")
    hierarchy.dendrogram(
        Z, labels=data.get_names(fingerprints[data.CLASS]), leaf_rotation=90.0
    )
    util.finish_plot("ott_fingerprint_dendogram", output_folder, show)


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(exists=True),
    help="Folder to save figures in",
)
@click.option("--show", "-s", is_flag=True, help="Show plots")
def main(path, output_folder, show):
    "Plot visualization of measurement file csv"
    measurements, _ = data.read_recursive(path)

    fingerprints = fingerprint.build_set(measurements, fingerprint.ott, add_class=True)
    _generate_heatmap(fingerprints, output_folder, show)
    _generate_pairplots(fingerprints, output_folder, show)

    _generate_dendogram(measurements, output_folder, show)
