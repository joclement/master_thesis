import click.testing

from thesis import measure_info


def test_measure_info_main_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(
        measure_info.main, ["./testdata/small_measurement_excerpt.csv"]
    )
    assert result.exit_code == 0


def test_measure_info_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--version"])
    assert result.exit_code == 0


def test_measure_info_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(measure_info.main, ["--help"])
    assert result.exit_code == 0
