from pathlib import Path

import click.testing

from thesis import classify


def test_classify_main_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [str(Path(csv_filepath).parent)])
    # note: Only 1 data item can not be split into training and test
    assert result.exit_code != 0


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
