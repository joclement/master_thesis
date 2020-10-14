import click

from . import __version__, measurement


def _echo_measurement_info(df):
    click.echo(df.describe())
    click.echo("")
    click.echo(df.info())
    click.echo("")
    click.echo(df.head(10))


@click.command()
@click.version_option(version=__version__)
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Print info recursively")
def main(path, recursive):
    """Print measurement info on given measurement file or folder

    PATH file or folder to print measurement info for
    """

    if not recursive:
        df = measurement.read(path)
        _echo_measurement_info(df)
    else:
        measurements = measurement.read_recursive(path)
        for df in measurements:
            _echo_measurement_info(df)
            click.echo(
                "\n ============================================================ \n"
            )
