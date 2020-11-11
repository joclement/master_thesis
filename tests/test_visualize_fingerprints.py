from pathlib import Path
from shutil import copyfile

import click.testing
import pytest

from thesis import visualize_fingerprints


@pytest.fixture
def folder_with_two_csv_files(csv_folder, tmpdir):
    for csv_file in list(Path(csv_folder).glob("*.csv"))[:2]:
        copyfile(csv_file, Path(tmpdir, csv_file.name))
    return str(tmpdir)


@pytest.mark.expensive
def test_visualize_fingerprints_main_with_save_succeeds(
    folder_with_two_csv_files, tmpdir
):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        visualize_fingerprints.main,
        ["--output-folder", tmpdir, folder_with_two_csv_files],
    )
    assert result.exit_code == 0
    assert len(list(Path(tmpdir).glob("*.png"))) == 5


def test_visualize_fingerprints_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_fingerprints.main, ["--version"])
    assert result.exit_code == 0


def test_visualize_fingerprints_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_fingerprints.main, ["--help"])
    assert result.exit_code == 0
