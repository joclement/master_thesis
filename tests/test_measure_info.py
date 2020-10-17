from pathlib import Path

import click.testing

from thesis import measure_info


def test_measure_info_main_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, [csv_filepath])
    assert result.exit_code == 0


def test_measure_info_main_recursive_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["-r", str(Path(csv_filepath).parent)])
    assert result.exit_code == 0


def test_measure_info_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--version"])
    assert result.exit_code == 0


def test_measure_info_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--help"])
    assert result.exit_code == 0
