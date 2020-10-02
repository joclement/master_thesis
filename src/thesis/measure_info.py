import click
from pandas_profiling import ProfileReport

from . import __version__, measurement

PROFILE_REPORT_PATH = "./report.html"


@click.command()
@click.version_option(version=__version__)
@click.argument("measurement_filepath", type=click.Path(exists=True))
def main(measurement_filepath):
    "Print info on measurement on given measurement file"

    df = measurement.read(measurement_filepath)

    click.echo(df.describe())

    click.echo("\n ============================================================ \n")
    click.echo(df.info())

    click.echo("\n ============================================================ \n")
    click.echo(df.head(10))

    click.echo("\n ============================================================ \n")
    click.echo(f"Save pandas profile report to {PROFILE_REPORT_PATH}")
    report = ProfileReport(df)
    report.to_file(output_file=PROFILE_REPORT_PATH)
