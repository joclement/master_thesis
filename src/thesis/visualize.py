import click
import matplotlib.pyplot as plt

from . import __version__, measurement


@click.command()
@click.version_option(version=__version__)
@click.argument("measurement_filepath", type=click.Path(exists=True))
@click.option("-o", "--output-folder", type=click.Path(exists=True))
@click.option("--show/--no-show", default=False)
def main(measurement_filepath, output_folder, show):
    "Plot visualization of measurement file csv"

    df = measurement.read(measurement_filepath)

    plt.scatter(df[measurement.TIME], df[measurement.PD], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("PD in nV")
    if output_folder:
        plt.savefig(f"{output_folder}/PDOverTime.png")
    if show:
        plt.show()

    plt.scatter(df[measurement.TIME][1:], df[measurement.TIMEDIFF][1:], marker=".")
    plt.xlabel("t in seconds")
    plt.ylabel("Δt in seconds")
    if output_folder:
        plt.savefig(f"{output_folder}/DeltaTOverTime.png")
    if show:
        plt.show()
