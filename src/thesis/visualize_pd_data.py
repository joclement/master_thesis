from pathlib import Path
from typing import List

import click
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__, data


def plot_pd_volts_over_time(df):
    plt.scatter(df[data.TIME], df[data.PD], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("PD in nV")


def plot_timediff_between_pds_over_time(df):
    plt.scatter(df[data.TIME][1:], df[data.TIMEDIFF][1:], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("Δt in seconds")


def plot_number_of_pds_over_time(df):
    bins = np.arange(0, df[data.TIME].max(), 1)
    counts = df.groupby(pd.cut(df[data.TIME], bins=bins)).size()
    fake = np.array([])
    for i in range(len(counts)):
        a, b = bins[i], bins[i + 1]
        sample = a + (b - a) * np.ones(counts[i])
        fake = np.append(fake, sample)
    plt.hist(fake, bins=bins)

    plt.xlabel("t in seconds")
    plt.ylabel("Number of PDs")


def plot_relation_between_consecutive_pd_volts(df):
    plt.hist2d(
        df[data.PD][:-1],
        df[data.PD][1:],
        100,
        cmin=1,
        cmap=plt.cm.jet,
        norm=colors.LogNorm(),
    )
    plt.colorbar()
    plt.xlabel("A(n) in nV")
    plt.ylabel("A(n+1) in nV")


def _generate_plots_for_single_csv(df: pd.DataFrame, output_folder, show):
    plot_pd_volts_over_time(df)
    if output_folder:
        plt.savefig(f"{output_folder}/PDOverTime.png")
    if show:
        plt.show()

    plot_timediff_between_pds_over_time(df)
    if output_folder:
        plt.savefig(f"{output_folder}/DeltaTOverTime.png")
    if show:
        plt.show()

    plot_number_of_pds_over_time(df)
    if output_folder:
        plt.savefig(f"{output_folder}/NumberOfPDsOverTime.png")
    if show:
        plt.show()

    plot_relation_between_consecutive_pd_volts(df)
    if output_folder:
        plt.savefig(f"{output_folder}/an-an+1.png")
    if show:
        plt.show()


def _generate_summary_plots_(measurements: List[pd.DataFrame], output_folder, show):
    _boxplot_lengths_of_pd_csvs_per_defect(measurements)
    if output_folder:
        plt.savefig(f"{output_folder}/boxplot_lengths_of_pd_csvs_per_defect.png")
    if show:
        plt.show()

    _plot_histogram_lengths_of_pd_csvs(measurements)
    if output_folder:
        plt.savefig(f"{output_folder}/histogram_lengths_of_pd_csvs.png")
    if show:
        plt.show()


def _boxplot_lengths_of_pd_csvs_per_defect(measurements):
    lengths_per_defect = {
        data.Defect(d): list() for d in set(data.get_defects(measurements))
    }
    for df in measurements:
        lengths_per_defect[data.Defect(df[data.CLASS][0])].append(len(df.index))
    fig, ax = plt.subplots()
    labels = [
        f"{data.DEFECT_NAMES[key]}: {len(value)}"
        for key, value in lengths_per_defect.items()
    ]
    ax.boxplot(lengths_per_defect.values(), labels=labels)
    plt.ylabel("Number of PDs in csv file")
    plt.xlabel("Defect type with number of samples")
    ax.set_title("Lengths of PD csv files")


def _plot_histogram_lengths_of_pd_csvs(measurements):
    lengths = [len(df.index) for df in measurements]
    fig, ax = plt.subplots()
    ax.hist(lengths, 10)
    plt.xlabel("Number of PDs in csv file")
    ax.set_title(f"Histogram of lengths of {len(lengths)} PD csv files")


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option("-o", "--output-folder", type=click.Path(exists=True))
@click.option("--show/--no-show", default=False)
def main(path, output_folder, show):
    "Plot visualization of measurement file csv"

    if Path(path).is_file():
        _generate_plots_for_single_csv(data.read(path), output_folder, show)
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        _generate_summary_plots_(measurements, output_folder, show)
