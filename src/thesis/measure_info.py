import click

from . import __version__, measurement


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
