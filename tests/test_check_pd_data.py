import click.testing

from jc_thesis_code import check_pd_data


def test_check_pd_data_main_file_succeeds(large_csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, [large_csv_filepath])
    assert result.exit_code == 0


def test_check_pd_data_main_file_expensive_succeeds(large_csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, ["-e", large_csv_filepath])
    assert result.exit_code == 0


def test_check_pd_data_main_folder_succeeds(tiny_csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, [tiny_csv_folder])
    assert result.exit_code == 0


def test_check_pd_data_main_folder_recursive_expensive_succeeds(tiny_csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, ["-r", "-e", tiny_csv_folder])
    assert result.exit_code == 0


def test_check_pd_data_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, ["--version"])
    assert result.exit_code == 0


def test_check_pd_data_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(check_pd_data.main, ["--help"])
    assert result.exit_code == 0
