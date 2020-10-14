import click.testing

from thesis import visualize


def test_visualize_main_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize.main, ["--no-show", csv_filepath])
    assert result.exit_code == 0


def test_visualize_main_save_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize.main,
        ["--output-folder", "/tmp/", csv_filepath],
    )
    assert result.exit_code == 0


def test_visualize_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize.main, ["--version"])
    assert result.exit_code == 0


def test_visualize_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize.main, ["--help"])
    assert result.exit_code == 0
