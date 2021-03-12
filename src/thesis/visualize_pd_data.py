from pathlib import Path
from typing import List

import click
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import __version__, data, prepared_data, util
from .data import CLASS, Defect, VOLTAGE_SIGN, VoltageSign


def _plot_pd_volts_over_time(df):
    plt.scatter(df[data.TIME_DIFF].cumsum(), df[data.PD], marker=".")
    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel("PD in nV")


def _plot_timediff_between_pds_over_time(df):
    plt.scatter(df[data.TIME_DIFF].cumsum(), df[data.TIME_DIFF], marker=".")
    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel(f"Δt {data.TIME_UNIT}")


def _plot_number_of_pds_over_time(df):
    bins = np.arange(0, df[data.TIME_DIFF].sum(), 1)
    counts = df.groupby(pd.cut(df[data.TIME_DIFF].cumsum(), bins=bins)).size()
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

    _stripplot_lengths_of_pd_csvs_per_defect(measurements)
    util.finish_plot("stripplot_lengths_of_pd_csv_per_defect", output_folder, show)

    _scatterplot_length_duration(measurements)
    util.finish_plot("scatterplot_lengths_durations", output_folder, show)

    _stripplot_duration_of_pd_csvs_per_defect(measurements)
    util.finish_plot("stripplot_duration_of_pd_csv_per_defect", output_folder, show)

    _plot_histogram_lengths_of_pd_csvs(measurements)
    util.finish_plot("histogram_lengths_of_pd_csvs", output_folder, show)

    _boxplot_duration_of_pd_csvs_per_defect(measurements)
    util.finish_plot("boxplot_duration_of_pd_csvs_per_defect", output_folder, show)

    _plot_histogram_duration_of_pd_csvs(measurements)
    util.finish_plot("histogram_duration_of_pd_csvs", output_folder, show)


_LENGTH_KEY = "Number of PDs x 1000"
_DURATION_KEY = "Duration [s]"


def _generate_polarity_plot(measurements: List[pd.DataFrame], output_folder, show):
    counts = {(str(defect), str(vs)): 0 for defect in Defect for vs in VoltageSign}
    for df in measurements:
        counts[(str(df.attrs[CLASS]), str(df.attrs[VOLTAGE_SIGN]))] += 1
    info_df = pd.DataFrame(
        data={
            "defect": [defect for defect, _ in counts.keys()],
            "polarity": [polarity for _, polarity in counts.keys()],
            "occurences": list(counts.values()),
        }
    )

    sns.barplot(x="defect", y="occurences", hue="polarity", data=info_df)
    util.finish_plot("occurences_of_polarity_per_defect", output_folder, show)


def _calc_duration_and_lengths(measurements):
    assert data.TIME_UNIT == "ms"
    rows = [
        {
            _LENGTH_KEY: len(df.index) / 1000,
            _DURATION_KEY: df[data.TIME_DIFF].sum() / 1000,
            data.CLASS: str(df.attrs[data.CLASS]),
        }
        for df in measurements
    ]
    return pd.DataFrame(rows)


def _boxplot_lengths_of_pd_csvs_per_defect(measurements):
    lengths_per_defect = {
        data.Defect(d): list() for d in set(data.get_defects(measurements))
    }
    for df in measurements:
        lengths_per_defect[data.Defect(df.attrs[data.CLASS])].append(len(df.index))
    fig, ax = plt.subplots()
    labels = [f"{str(key)}: {len(value)}" for key, value in lengths_per_defect.items()]
    ax.boxplot(lengths_per_defect.values(), labels=labels)
    plt.ylabel("Number of PDs in csv file")
    plt.xlabel("Defect type with number of samples")
    ax.set_title(f"Lengths of {len(measurements)} PD csv files")


def _stripplot_lengths_of_pd_csvs_per_defect(measurements):
    sns.stripplot(
        data=_calc_duration_and_lengths(measurements), x=_LENGTH_KEY, y=data.CLASS
    )


def _scatterplot_length_duration(measurements):
    sns.relplot(
        data=_calc_duration_and_lengths(measurements),
        x=_LENGTH_KEY,
        y=_DURATION_KEY,
        hue=data.CLASS,
    )


def _stripplot_duration_of_pd_csvs_per_defect(measurements):
    sns.stripplot(
        data=_calc_duration_and_lengths(measurements), x=_DURATION_KEY, y=data.CLASS
    )


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
        duration_per_defect[data.Defect(df.attrs[data.CLASS])].append(
            df[data.TIME_DIFF].sum()
        )
    fig, ax = plt.subplots()
    labels = [f"{str(key)}: {len(value)}" for key, value in duration_per_defect.items()]
    ax.boxplot(duration_per_defect.values(), labels=labels)
    plt.ylabel(f"Duration {data.TIME_UNIT}")
    plt.xlabel("Defect type with number of samples")
    ax.set_title(f"Duration of {len(measurements)} PD csv files")


def _plot_histogram_duration_of_pd_csvs(measurements):
    durations = [df[data.TIME_DIFF].sum() for df in measurements]
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
@click.option("--split", "-s", is_flag=True, help="Split data into 60 seconds samples")
def main(path, output_folder, recursive, show, split):
    "Plot visualization of measurement file csv"

    if Path(path).is_file():
        _generate_plots_for_single_csv(data.read(path), output_folder, show)
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        if split and not recursive:
            measurements = prepared_data.adapt_durations(measurements)
        if recursive:
            for measurement, csv_filepath in zip(measurements, csv_filepaths):
                single_csv_folder = Path(output_folder, Path(csv_filepath).name)
                single_csv_folder.mkdir(parents=True, exist_ok=False)
                _generate_plots_for_single_csv(measurement, single_csv_folder, show)
        _generate_summary_plots(measurements, output_folder, show)
        _generate_polarity_plot(measurements, output_folder, show)
