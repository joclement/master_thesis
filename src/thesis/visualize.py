import click

from thesis import measurement

from . import __version__


@click.command()
@click.version_option(version=__version__)
@click.argument("measurement_filepath", type=click.Path(exists=True))
def main(measurement_filepath):
    "Plot visualization of measurement file csv"

    df = measurement.read(measurement_filepath)
    click.echo(df.head(10))
