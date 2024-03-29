import filecmp
from itertools import combinations
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd

from . import __version__, data, fingerprint


def _echo_measurement_info(df):
    click.echo(df.describe())
    click.echo("")
    click.echo(df.info())
    click.echo("")
    click.echo(df.head(10))
    click.echo("")
    if df[data.PD].min() < 0.0:
        click.echo("This experiment contains negative PD values.")


def _echo_fingerprint_info(df):
    click.echo("Fingerprint TU Graz:")
    click.echo(fingerprint.tugraz(df))
    click.echo("")
    click.echo("Fingerprint Lukas:")
    click.echo(fingerprint.ott(df))


def _ensure_unique(csv_filepaths: list):
    for file1, file2 in combinations(csv_filepaths, 2):
        if filecmp.cmp(file1, file2):
            raise ValueError(f"There are duplicates: '{file1}' and '{file2}'")


def _info_on_unique_filenames(csv_filepaths: list):
    click.echo("Checking for duplicate filenames")
    for file1, file2 in combinations(csv_filepaths, 2):
        file1 = Path(file1).name.replace(" ", "")
        file2 = Path(file2).name.replace(" ", "")
        if file1 == file2:
            raise ValueError(
                f"Duplicate filenames: '{file1}' and '{file2}' !!!!!!!!!!!!!!!!!"
            )


def _is_subarray(a, b):
    try:
        i = np.where(a == b[0])[0][0]
    except IndexError:
        return b.size == 0
    a = a[i : i + b.size]
    if a.size < b.size:
        return False
    return (a == b).all()


def _check_subarrays(df1, df2):
    if _is_subarray(df2[data.PD].values, df1[data.PD].values):
        click.echo("df1 PD values are sub array of df2 PD values!")
    if _is_subarray(df2[data.TIME_DIFF].values, df1[data.TIME_DIFF].values):
        click.echo("df1 time diff values are sub array of df2 time diff values!")
    if _is_subarray(df1[data.PD].values, df2[data.PD].values):
        click.echo("df2 PD values are sub array of df1 PD values!")
    if _is_subarray(df1[data.TIME_DIFF].values, df2[data.TIME_DIFF].values):
        click.echo("df2 time diff values are sub array of df1 time diff values!")


def _check_partly_duplicates(measurements: list, csv_filepaths: list):
    for fm1, fm2 in combinations(zip(measurements, csv_filepaths), 2):
        df1, file1 = fm1
        df2, file2 = fm2
        file1 = Path(file1)
        file2 = Path(file2)

        if df1.isin(df2).all().all() or df2.isin(df1).all().all():
            click.echo(f"filenames:\n df1: '{str(file1)}'\n df2: '{str(file2)}'")
            click.echo(f"df1 is in df2: {df1.isin(df2).all().all()}")
            click.echo(f"df2 is in df1: {df2.isin(df1).all().all()}")
            _check_subarrays(df1, df2)
            raise ValueError("Likely possible duplicate content!")
        if df1.isin(df2).all().any() or df2.isin(df1).all().any():
            click.echo("Maybe possible duplicate content:")
            click.echo(f"filenames:\n df1: '{str(file1)}'\n df2: '{str(file2)}'")
            click.echo(f"df1 is in df2: {df1.isin(df2).all().any()}")
            click.echo(f"df2 is in df1: {df2.isin(df1).all().any()}")
            _check_subarrays(df1, df2)


def _check_time_starts_with_zero(measurements: List[pd.DataFrame]):
    click.echo("Report files starting not with 0 time:")
    for df in measurements:
        if df.attrs[data.START_TIME] > 0:
            click.echo(f"    {df.attrs[data.PATH]}")


def _info_on_negative_pds(measurements, csv_filepaths):
    files_with_neg_pd_values = [
        csv for df, csv in zip(measurements, csv_filepaths) if df[data.PD].min() < 0.0
    ]
    if len(files_with_neg_pd_values) == 0:
        click.echo("No files have negative values.")
    else:
        click.echo("Summary on files with negative values:")
        for path in files_with_neg_pd_values:
            click.echo(path)


def _info_on_too_few_pds_per_sec(measurements, csv_filepaths):
    too_few_pds_per_sec_files = {}
    for df, csv in zip(measurements, csv_filepaths):
        too_few_pds_per_sec_files[csv] = 0
        for idx in range(df[data.PD].size - 2):
            if df[data.TIME_DIFF][idx : idx + 3].sum() > 30000.0:
                too_few_pds_per_sec_files[csv] += 1

    if (np.array(list(too_few_pds_per_sec_files.values())) == 0).all():
        click.echo("No files have less than 3 PDs per 30 seconds.")
    else:
        click.echo("Summary on files with less than 3 PDs per 30 seconds:")
        for csv_path, times in too_few_pds_per_sec_files.items():
            if times > 0:
                click.echo(f"{csv_path} with {times} times")


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Print detailed info on every file in folder",
)
@click.option(
    "--expensive",
    "-e",
    is_flag=True,
    help="Perform expensive computations: Check for enough PDs",
)
def main(path, recursive, expensive):
    """
    Print measurement info on given PD measurement csv file or folder containing csv
    files.

    PATH  existing file or folder to check pd data in
    """

    if Path(path).is_file():
        df = data.read(path)
        _echo_measurement_info(df)
        click.echo("")
        _echo_fingerprint_info(df)
        if expensive:
            _info_on_too_few_pds_per_sec([df], [path])
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        _ensure_unique(csv_filepaths)
        if recursive:
            for df, csv_filepath in zip(measurements, csv_filepaths):
                click.echo(f"Info on: '{csv_filepath}'")
                _echo_measurement_info(df)
                click.echo("")
                _echo_fingerprint_info(df)
                click.echo(
                    "\n ============================================================ \n"
                )

        min_pd = min([measurement[data.PD].min() for measurement in measurements])
        max_pd = max([measurement[data.PD].max() for measurement in measurements])
        min_timediff = min(
            [measurement[data.TIME_DIFF].min() for measurement in measurements]
        )
        max_timediff = max(
            [measurement[data.TIME_DIFF].max() for measurement in measurements]
        )

        click.echo(f"Overall min PD value: {min_pd}")
        click.echo(f"Overall max PD value: {max_pd}")
        click.echo(f"Overall min TimeDiff value: {min_timediff}")
        click.echo(f"Overall max TimeDiff value: {max_timediff}")

        count_below_60 = sum(df[data.TIME_DIFF].sum() < 60000 for df in measurements)
        click.echo(f"Measurements shorter than 60 seconds: {count_below_60}")

        _info_on_negative_pds(measurements, csv_filepaths)

        if expensive:
            _info_on_too_few_pds_per_sec(measurements, csv_filepaths)

        _check_time_starts_with_zero(measurements)

        _info_on_unique_filenames(csv_filepaths)
        _check_partly_duplicates(measurements, csv_filepaths)

        click.echo(f"Min len: {min([len(df.index) for df in measurements])}")
        click.echo(f"Max len: {max([len(df.index) for df in measurements])}")
