import click.testing
import pandas as pd

from thesis import tsfresh_features


def test_main(csv_folder):
    relevance_table = tsfresh_features.calc_relevant_features(csv_folder, n_jobs=1)
    assert type(relevance_table) is pd.DataFrame


def test_convert_to_tsfresh_dataset(measurements):
    dataset = tsfresh_features.convert_to_tsfresh_dataset(measurements)
    assert type(dataset) is pd.DataFrame


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(tsfresh_features.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(tsfresh_features.main, ["--help"])
    assert result.exit_code == 0
