import filecmp
from itertools import combinations

import click

from . import __version__, data, fingerprint


def _echo_measurement_info(df):
    click.echo(df.describe())
    click.echo("")
    click.echo(df.info())
    click.echo("")
    click.echo(df.head(10))


def _echo_fingerprint_info(df):
    click.echo(fingerprint.tu_graz(df))


def _ensure_unique(csv_filepaths: list):
    for file1, file2 in combinations(csv_filepaths, 2):
        if filecmp.cmp(file1, file2):
            raise ValueError(f"There are duplicates: '{file1}' and '{file2}'")


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Print info recursively")
def main(path, recursive):
    """Print measurement info on given measurement file or folder

    PATH file or folder to print measurement info for
    """

    if not recursive:
        df = data.read(path)
        _echo_measurement_info(df)
        click.echo("")
        _echo_fingerprint_info(df)
    else:
        measurements, csv_filepaths = data.read_recursive(path)
        _ensure_unique(csv_filepaths)
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
            [measurement[data.TIMEDIFF].min() for measurement in measurements]
        )
        max_timediff = max(
            [measurement[data.TIMEDIFF].max() for measurement in measurements]
        )

        click.echo(f"Overall min PD value: {min_pd}")
        click.echo(f"Overall max PD value: {max_pd}")
        click.echo(f"Overall min TimeDiff value: {min_timediff}")
        click.echo(f"Overall max TimeDiff value: {max_timediff}")
