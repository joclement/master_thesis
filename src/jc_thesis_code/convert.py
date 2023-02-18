from pathlib import Path

import click
import pandas as pd
import scipy.io

from . import __version__, data


@click.command()
@click.version_option(version=__version__)
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def mat2csv(input_path, output_path):
    """Convert MAT file(s) to CSV file(s)

    INPUT_PATH   path to file or folder to read MAT file(s) from

    OUTPUT_PATH  path to file or folder to write CSV file(s) to
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_file():
        if output_path.exists():
            raise ValueError(f"Error: OUTPUT_PATH {output_path} exists.")
        _matfile2csvfile(input_path, output_path)
    else:
        if not output_path.is_dir():
            raise ValueError(f"Error: OUTPUT_PATH {output_path} is not a directory.")
        for input_file in Path(input_path).rglob("*.mat"):
            output_file = Path(output_path, input_file.relative_to(input_path))
            output_file = output_file.with_suffix(".csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            _matfile2csvfile(input_file, output_file)


def _flatten(list_of_sublists: list) -> list:
    return [item for sublist in list_of_sublists for item in sublist]


def _matfile2csvfile(mat_filepath, csv_filepath):
    click.echo(f"convert {mat_filepath} to {csv_filepath}")
    mat = scipy.io.loadmat(mat_filepath)
    mat = {k: v for k, v in mat.items() if k[0] != "_"}
    if all([len(entry) == 1 for _, value in mat.items() for entry in value]):
        mat = {key: _flatten(value) for key, value in mat.items()}
    else:
        raise ValueError("MAT file not convertable")
    measurement = pd.DataFrame({k: pd.Series(v) for k, v in mat.items()})
    measurement = measurement.rename(columns={"tm": data.TIME_IN_FILE, "V": data.PD})
    # @note: The PD data is in V and will be converted to nV
    measurement[data.PD] *= 10 ** 9
    measurement.to_csv(
        csv_filepath,
        sep=data.SEPERATOR,
        columns=[data.TIME_IN_FILE, data.PD],
        index=False,
        decimal=data.DECIMAL_SIGN,
    )
