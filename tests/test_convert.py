import csv
from pathlib import Path

import click.testing
import pytest

from thesis import convert, measurement


@pytest.fixture
def mat_filepath():
    return Path("./testdata/measurement_sample.mat")


def test_convert_mat2csv(mat_filepath, tmp_path):
    csv_filepath = Path(tmp_path, "measurement_sample.csv")
    assert not csv_filepath.exists()
    runner = click.testing.CliRunner()
    result = runner.invoke(convert.mat2csv, [f"{mat_filepath}", f"{csv_filepath}"])
    assert result.exit_code == 0
    assert csv_filepath.exists()
    with open(csv_filepath, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=measurement.SEPERATOR)
        header = next(reader)
        assert len(header) == 2
        assert header == [measurement.TIME, measurement.PD]
        for row in reader:
            assert len(row) == 2


def test_convert_mat2csv_recursive(mat_filepath, tmp_path):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        convert.mat2csv, ["-r", f"{mat_filepath.parent}", f"{tmp_path}"]
    )
    assert result.exit_code == 0
    csv_filepath = Path(tmp_path, mat_filepath.name).with_suffix(".csv")
    assert csv_filepath.exists()


def test_convert_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(convert.mat2csv, ["--version"])
    assert result.exit_code == 0


def test_convert_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(convert.mat2csv, ["--help"])
    assert result.exit_code == 0
