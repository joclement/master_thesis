from pathlib import Path
from typing import List

import click
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__, data, util


def _plot_pd_volts_over_time(df):
    plt.scatter(df[data.TIME], df[data.PD], marker=".")
    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel("PD in nV")


def _plot_timediff_between_pds_over_time(df):
    plt.scatter(df[data.TIME][1:], df[data.TIMEDIFF][1:], marker=".")
    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel(f"Δt {data.TIME_UNIT}")


def _plot_number_of_pds_over_time(df):
    bins = np.arange(0, df[data.TIME].max(), 1)
    counts = df.groupby(pd.cut(df[data.TIME], bins=bins)).size()
    fake = np.array([])
    for i in range(len(counts)):
        a, b = bins[i], bins[i + 1]
        sample = a + (b - a) * np.ones(counts[i])
        fake = np.append(fake, sample)
    plt.hist(fake, bins=bins)

    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel("Number of PDs")


def _plot_relation_between_consecutive_pd_volts(df):
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
    _plot_pd_volts_over_time(df)
    util.finish_plot("PDOverTime", output_folder, show)

    _plot_timediff_between_pds_over_time(df)
    util.finish_plot("DeltaTOverTime", output_folder, show)

    _plot_number_of_pds_over_time(df)
    util.finish_plot("NumberOfPDsOverTime", output_folder, show)

    _plot_relation_between_consecutive_pd_volts(df)
    util.finish_plot("an-an+1", output_folder, show)


def _generate_summary_plots(measurements: List[pd.DataFrame], output_folder, show):
    _boxplot_lengths_of_pd_csvs_per_defect(measurements)
    util.finish_plot("boxplot_lengths_of_pd_csvs_per_defect", output_folder, show)

    _plot_histogram_lengths_of_pd_csvs(measurements)
    util.finish_plot("histogram_lengths_of_pd_csvs", output_folder, show)

    _boxplot_duration_of_pd_csvs_per_defect(measurements)
    util.finish_plot("boxplot_duration_of_pd_csvs_per_defect", output_folder, show)

    _plot_histogram_duration_of_pd_csvs(measurements)
    util.finish_plot("histogram_duration_of_pd_csvs", output_folder, show)


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
    ax.set_title(f"Lengths of {len(measurements)} PD csv files")


def _plot_histogram_lengths_of_pd_csvs(measurements):
    lengths = [len(df.index) for df in measurements]
    fig, ax = plt.subplots()
    ax.hist(lengths, 10)
    plt.xlabel("Number of PDs in csv file")
    ax.set_title(f"Histogram of lengths of {len(measurements)} PD csv files")


def _boxplot_duration_of_pd_csvs_per_defect(measurements):
    duration_per_defect = {
        data.Defect(d): list() for d in set(data.get_defects(measurements))
    }
    for df in measurements:
        duration_per_defect[data.Defect(df[data.CLASS][0])].append(df[data.TIME].max())
    fig, ax = plt.subplots()
    labels = [
        f"{data.DEFECT_NAMES[key]}: {len(value)}"
        for key, value in duration_per_defect.items()
    ]
    ax.boxplot(duration_per_defect.values(), labels=labels)
    plt.ylabel(f"Duration {data.TIME_UNIT}")
    plt.xlabel("Defect type with number of samples")
    ax.set_title(f"Duration of {len(measurements)} PD csv files")


def _plot_histogram_duration_of_pd_csvs(measurements):
    durations = [df[data.TIME].max() for df in measurements]
    fig, ax = plt.subplots()
    ax.hist(durations, 10)
    plt.xlabel(f"Duration {data.TIME_UNIT}")
    ax.set_title(f"Histogram of duration of {len(measurements)} PD csv files")


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(exists=True),
    help="Folder to save figures in",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Generate plots for every file in folder",
)
@click.option("--show", "-s", is_flag=True, help="Show plots")
def main(path, output_folder, recursive, show):
    "Plot visualization of measurement file csv"

    if Path(path).is_file():
        _generate_plots_for_single_csv(data.read(path), output_folder, show)
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        if recursive:
            for measurement in measurements:
                _generate_plots_for_single_csv(measurement, output_folder, show)
        _generate_summary_plots(measurements, output_folder, show)
