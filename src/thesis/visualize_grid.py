import pickle

import click

from . import __version__


@click.command()
@click.version_option(version=__version__)
@click.argument("grid_file", type=click.Path(exists=True))
def main(grid_file):
    with open(grid_file, "rb") as f:
        grid_results = pickle.load(f)
    print(grid_results)
