import pandas as pd
import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
@click.argument("measurement_filepath", type=click.Path(exists=True))
def main(measurement_filepath):
    "Plot visualization of measurement file csv"

    df = pd.read_csv(measurement_filepath, sep=";")
    click.echo(df.head(10))
