from random import shuffle

import click
import pandas as pd
from scipy.cluster import hierarchy
import seaborn as sns

from . import __version__, data, fingerprint, prepared_data, util


def _generate_heatmap(fingerprints: pd.DataFrame, output_folder, show):
    corr = fingerprints.corr()
    sns.heatmap(corr, center=0)
    util.finish_plot("correlation_fingerprint", output_folder, show)


def _generate_pairplots(fingerprints: pd.DataFrame, output_folder, use_groups, show):
    fingerprints[data.CLASS] = data.get_names(fingerprints[data.CLASS])
    if use_groups:
        for group in fingerprint.Group:
            features = fingerprint.get_parameter_group(fingerprints, group)
            features[data.CLASS] = data.get_names(fingerprints[data.CLASS])
            pairgrid = sns.pairplot(features, hue=data.CLASS)
            pairgrid.fig.suptitle(
                f"Pairwise relationships in fingerprint {group} parameters",
            )
            util.finish_plot(f"pairplot_{group}", output_folder, show)
    else:
        features = list(fingerprints.columns)
        shuffle(features)
        for idx_part_start in range(0, len(features), 8):
            pairgrid = sns.pairplot(
                fingerprints[
                    set(features[idx_part_start : idx_part_start + 8] + [data.CLASS])
                ],
                hue=data.CLASS,
            )
            pairgrid.fig.suptitle("Pairwise relationships of 8 features")
            util.finish_plot(f"pairplot_{idx_part_start}", output_folder, show)


def _generate_dendogram(fingerprints: pd.DataFrame, output_folder, show):
    Z = hierarchy.linkage(fingerprints.drop(data.CLASS, axis=1), "single")
    hierarchy.dendrogram(
        Z, labels=data.get_names(fingerprints[data.CLASS]), leaf_rotation=90.0
    )
    util.finish_plot("fingerprint_dendogram", output_folder, show)


@click.command()
@click.version_option(version=__version__)
@click.argument("finger", type=str)
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(exists=True),
    help="Folder to save figures in",
)
@click.option("--show", "-s", is_flag=True, help="Show plots")
@click.option("--split", "-s", is_flag=True, help="Split data into 60 seconds samples")
@click.option("--group", "-g", is_flag=True, help="Use groups for pairplots")
def main(finger, path, output_folder, show, split, group):
    "Plot visualization of measurement file csv"
    measurements, _ = data.read_recursive(path, data.TreatNegValues.absolute)
    if split:
        measurements = prepared_data.adapt_durations(
            measurements, max_duration="60 seconds"
        )

    fingerprints = fingerprint.build_set(
        measurements, getattr(fingerprint, finger), add_class=True
    )
    _generate_heatmap(fingerprints, output_folder, show)
    _generate_pairplots(fingerprints, output_folder, group, show)

    _generate_dendogram(fingerprints, output_folder, show)
