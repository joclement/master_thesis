import click.testing

from thesis import measure_info


def test_measure_info_main_succeeds(large_csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["-e", large_csv_filepath])
    assert result.exit_code == 0


def test_measure_info_main_recursive_succeeds(csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["-r", "-e", csv_folder])
    assert result.exit_code == 0


def test_measure_info_main_verbose_succeeds(csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["-r", "-v", csv_folder])
    assert result.exit_code == 0


def test_measure_info_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--version"])
    assert result.exit_code == 0


def test_measure_info_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--help"])
    assert result.exit_code == 0
