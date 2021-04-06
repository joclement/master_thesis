from pathlib import Path

import click.testing
import pandas as pd

from thesis import tsfresh_features


def test_main_with_features_saved(csv_folder, tmpdir):
    output_file = Path(tmpdir, "extracted_features.data")
    assert not output_file.exists()
    runner = click.testing.CliRunner()
    result = runner.invoke(
        tsfresh_features.main,
        [
            csv_folder,
            "-j",
            1,
            "-o",
            str(output_file),
            "-d",
            "223 seconds",
            "-f",
            "500ms",
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(tsfresh_features.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(tsfresh_features.main, ["--help"])
    assert result.exit_code == 0


def test__convert_to_tsfresh_dataset(measurements):
    dataset = tsfresh_features._convert_to_tsfresh_dataset(
        measurements, "223 seconds", "50ms"
    )
    assert type(dataset) is pd.DataFrame
    assert dataset["value"].dtype == "float32"
    assert dataset["id"].dtype == "int16"
    assert dataset["sort"].dtype == "int32"
