import click.testing

from thesis import visualize


def test_main_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize.main, ["./testdata/small_measurement_excerpt.csv"])
    assert result.exit_code == 0
