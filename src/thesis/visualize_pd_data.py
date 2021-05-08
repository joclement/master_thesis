import math
from pathlib import Path
from typing import List

import click
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import __version__, data, prepared_data, util
from .data import CLASS, VOLTAGE_SIGN, VoltageSign


def _plot_pd_volts_over_time(df):
    plt.scatter(df[data.TIME_DIFF].cumsum(), df[data.PD], marker=".")
    plt.xlabel(f"t {data.TIME_UNIT}")
    plt.ylabel("PD in nV")


def plot_sliding_window(df):
    ts = df.copy(deep=True)
    assert data.TIME_UNIT == "ms"
    ts[data.TIME_DIFF] /= 1000
    ts["time"] = ts[data.TIME_DIFF].cumsum()
    ts = ts.drop(np.random.choice(ts.index, int(8 * len(ts.index) / 10), replace=False))

    plt.scatter(ts["time"], ts[data.PD], marker=".", s=0.2, rasterized=True)
    plt.xlabel("Time (sec)")
    plt.ylabel("PD (nV)")

    length = 60
    x_step = 30
    windows = list(range(0, math.ceil(ts["time"].iloc[-1]) - length, x_step))
    ys = [i / (len(windows) - 0.9) * ts[data.PD].iloc[-1] for i in range(len(windows))]
    xmins = [i for i in windows]
    xmaxs = [i + length for i in windows]
    plt.hlines(ys, xmin=xmins, xmax=xmaxs, colors="black")
    anno_args = {"ha": "center", "va": "center", "size": 24, "color": "black"}
    for xmin, xmax, y in zip(xmins, xmaxs, ys):
        plt.annotate("|", xy=(xmin, y), **anno_args)
        plt.annotate("|", xy=(xmax, y), **anno_args)

    cmap = plt.cm.Reds
    for idx, val in enumerate(windows):
        plt.axvspan(val, val + length, alpha=0.3, color=cmap(0.75 - idx / len(windows)))

    plt.text(
        xmins[0] + x_step,
        ys[0],
        "Window size",
        verticalalignment="bottom",
        horizontalalignment="center",
    )

    step_window_y = 0.45 * (ys[1] - ys[0]) + ys[0]
    plt.hlines(step_window_y, xmin=xmins[0], xmax=xmins[0] + x_step, colors="dimgray")
    anno_args = {"ha": "center", "va": "center", "size": 20, "color": "dimgray"}
    plt.annotate("|", xy=(xmins[0], step_window_y), **anno_args)
    plt.annotate("|", xy=(xmins[0] + x_step, step_window_y), **anno_args)
    plt.text(
        xmins[0] + x_step / 2,
        step_window_y,
        "Step size",
        verticalalignment="bottom",
        horizontalalignment="center",
        color="dimgray",
    )


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
    util.finish_plot("boxplots_lengths_of_pd_csv_per_defect", output_folder, show)

    _stripplot_lengths_of_pd_csvs_per_defect(measurements)
    util.finish_plot("stripplot_lengths_of_pd_csv_per_defect", output_folder, show)

    _scatterplot_length_duration(measurements)
    util.finish_plot("scatterplot_lengths_durations", output_folder, show)

    _stripplot_duration_of_pd_csvs_per_defect(measurements)
    util.finish_plot("stripplot_duration_of_pd_csv_per_defect", output_folder, show)


_LENGTH_KEY = "Number of PDs x 1000"
_DURATION_KEY = "Duration [s]"


def _generate_polarity_plot(measurements: List[pd.DataFrame], output_folder, show):
    defects = set(data.get_defects(measurements))
    counts = {(defect, str(vs)): 0 for defect in defects for vs in VoltageSign}
    for df in measurements:
        counts[(df.attrs[CLASS], str(df.attrs[VOLTAGE_SIGN]))] += 1
    info_df = pd.DataFrame(
        data={
            "defect": [defect.wrapped() for defect, _ in counts.keys()],
            "polarity": [polarity for _, polarity in counts.keys()],
            "occurence": list(counts.values()),
        }
    )

    sns.barplot(x="defect", y="occurence", hue="polarity", data=info_df)
    util.finish_plot("occurence_of_polarity_per_defect", output_folder, show)


def _calc_duration_and_lengths(measurements):
    assert data.TIME_UNIT == "ms"
    rows = [
        {
            _LENGTH_KEY: len(df.index) / 1000,
            _DURATION_KEY: df[data.TIME_DIFF].sum() / 1000,
            data.CLASS: df.attrs[data.CLASS].wrapped(),
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
    plt.ylabel("Number of PDs")
    plt.xlabel("Defect type with number of samples")


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
    plt.xlabel("Number of PDs")


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


def _plot_histogram_duration_of_pd_csvs(measurements):
    durations = [df[data.TIME_DIFF].sum() for df in measurements]
    fig, ax = plt.subplots()
    ax.hist(durations, 10)
    plt.xlabel(f"Duration {data.TIME_UNIT}")


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

    sns.set(font_scale=1.2)

    if Path(path).is_file():
        _generate_plots_for_single_csv(data.read(path), output_folder, show)
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        if recursive:
            for measurement, csv_filepath in zip(measurements, csv_filepaths):
                single_csv_folder = Path(output_folder, Path(csv_filepath).name)
                single_csv_folder.mkdir(parents=True, exist_ok=False)
                _generate_plots_for_single_csv(measurement, single_csv_folder, show)
        if split:
            measurements = prepared_data.adapt_durations(
                measurements, max_duration="60 seconds"
            )
        _generate_summary_plots(measurements, output_folder, show)
        _generate_polarity_plot(measurements, output_folder, show)
