import click.testing

from thesis import visualize_pd_data


def test_visualize_pd_data_main_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--no-show", csv_filepath])
    assert result.exit_code == 0


def test_visualize_pd_data_main_save_succeeds(csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_pd_data.main,
        ["--output-folder", "/tmp/", csv_filepath],
    )
    assert result.exit_code == 0


def test_visualize_pd_data_main_dir_succeeds(csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--no-show", csv_folder])
    assert result.exit_code == 0


def test_visualize_pd_data_main_dir_save_succeeds(csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_pd_data.main,
        ["--output-folder", "/tmp/", csv_folder],
    )
    assert result.exit_code == 0


def test_visualize_pd_data_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--version"])
    assert result.exit_code == 0


def test_visualize_pd_data_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--help"])
    assert result.exit_code == 0
