from typing import List

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import __version__, data, fingerprint, util


def _generate_fingerprint_plots(measurements: List[pd.DataFrame], output_folder, show):
    fingerprints = fingerprint.build_set(measurements, fingerprint.lukas_plus_tu_graz)
    corr = fingerprints.corr()
    sns.heatmap(corr, center=0)
    util.finish_plot("correlation_fingerprint_lukas+tu_graz", output_folder, show)

    for group in fingerprint.Group:
        parameters = fingerprint.get_parameter_group(fingerprints, group)
        parameters[data.CLASS] = fingerprints[data.CLASS]

        sns.pairplot(parameters, hue=data.CLASS)
        plt.title(
            f"Pairwise relationships in fingerprint parameters related to {group.value}"
        )
        util.finish_plot(f"pairplot_{group.value}", output_folder, show)


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output-folder", type=click.Path(exists=True))
@click.option("--show", "-s", is_flag=True, help="Show plots")
def main(path, output_folder, show):
    "Plot visualization of measurement file csv"
    measurements, _ = data.read_recursive(path)
    _generate_fingerprint_plots(measurements, output_folder, show)