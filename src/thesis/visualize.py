import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__, measurement


def plot_number_of_pds_over_time(plt, df):
    bins = np.arange(0, df[measurement.TIME].max(), 1)
    counts = df.groupby(pd.cut(df[measurement.TIME], bins=bins)).size()
    fake = np.array([])
    for i in range(len(counts)):
        a, b = bins[i], bins[i + 1]
        sample = a + (b - a) * np.ones(counts[i])
        fake = np.append(fake, sample)
    plt.hist(fake, bins=bins)

    plt.xlabel("t in seconds")
    plt.ylabel("Number of PDs")


@click.command()
@click.version_option(version=__version__)
@click.argument("measurement_filepath", type=click.Path(exists=True))
@click.option("-o", "--output-folder", type=click.Path(exists=True))
@click.option("--show/--no-show", default=False)
def main(measurement_filepath, output_folder, show):
    "Plot visualization of measurement file csv"

    df = measurement.read(measurement_filepath)

    plt.scatter(df[measurement.TIME], df[measurement.PD], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("PD in nV")
    if output_folder:
        plt.savefig(f"{output_folder}/PDOverTime.png")
    if show:
        plt.show()

    plt.scatter(df[measurement.TIME][1:], df[measurement.TIMEDIFF][1:], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("Δt in seconds")
    if output_folder:
        plt.savefig(f"{output_folder}/DeltaTOverTime.png")
    if show:
        plt.show()

    plot_number_of_pds_over_time(plt, df)
    if output_folder:
        plt.savefig(f"{output_folder}/NumberOfPDsOverTime.png")
    if show:
        plt.show()
