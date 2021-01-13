from pathlib import Path

import click.testing
import pandas as pd

from thesis import tsfresh_features


def test_main_with_features_saved(csv_folder, tmpdir):
    output_file = Path(tmpdir, "extracted_features.data")
    assert not output_file.exists()
    runner = click.testing.CliRunner()
    result = runner.invoke(tsfresh_features.main, [csv_folder, "-j", 1, "-o", str(output_file)])
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
