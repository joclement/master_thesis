from pathlib import Path

import click.testing

from thesis import visualize_pd_data


def test_visualize_pd_data_main_succeeds(tiny_csv_filepath):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, [tiny_csv_filepath])
    assert result.exit_code == 0


def test_visualize_pd_data_main_save_succeeds(tiny_csv_filepath, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_pd_data.main,
        ["--output-folder", tmpdir, tiny_csv_filepath],
    )
    assert result.exit_code == 0
    assert len(list(Path(tmpdir).glob("*.svg"))) == 4


def test_visualize_pd_data_main_dir_succeeds(tiny_csv_folder):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, [tiny_csv_folder])
    assert result.exit_code == 0


def test_visualize_pd_data_main_dir_save_succeeds(tiny_csv_folder, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_pd_data.main,
        ["--output-folder", tmpdir, tiny_csv_folder],
    )
    assert result.exit_code == 0
    assert len(list(Path(tmpdir).glob("*.svg"))) == 7


def test_visualize_pd_data_main_dir_save_recursive_succeeds(tiny_csv_folder, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_pd_data.main,
        ["-r", "--output-folder", tmpdir, tiny_csv_folder],
    )
    assert result.exit_code == 0
    assert sum(sub_path.is_dir() for sub_path in Path(tmpdir).iterdir()) == 2


def test_visualize_pd_data_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--version"])
    assert result.exit_code == 0


def test_visualize_pd_data_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_pd_data.main, ["--help"])
    assert result.exit_code == 0
