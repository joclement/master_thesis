from pathlib import Path
from shutil import copyfile
from typing import List

import click.testing
import pandas as pd
import pytest

from thesis import data, fingerprint, visualize_fingerprints


@pytest.fixture
def folder_with_two_csv_files(csv_folder, tmpdir):
    for csv_file in list(Path(csv_folder).glob("*.csv"))[:2]:
        copyfile(csv_file, Path(tmpdir, csv_file.name))
    return str(tmpdir)


@pytest.fixture
def two_measurements(folder_with_two_csv_files) -> List[pd.DataFrame]:
    return data.read_recursive(folder_with_two_csv_files)[0]


@pytest.fixture
def fingerprints(two_measurements):
    return fingerprint.build_set(two_measurements, fingerprint.ott, add_class=True)


def test__generate_heatmap(fingerprints, tmpdir):
    visualize_fingerprints._generate_heatmap(fingerprints, tmpdir, False)
    assert len(list(Path(tmpdir).glob("*.svg"))) == 1


def test__generate_dendogram(fingerprints, tmpdir):
    visualize_fingerprints._generate_dendogram(fingerprints, tmpdir, False)
    assert len(list(Path(tmpdir).glob("*.svg"))) == 1


@pytest.mark.expensive
def test__generate_pairplots(fingerprints, tmpdir):
    visualize_fingerprints._generate_pairplots(fingerprints, tmpdir, False)
    assert len(list(Path(tmpdir).glob("*.svg"))) == 4


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
    assert len(list(Path(tmpdir).glob("*.svg"))) == 6


def test_visualize_fingerprints_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_fingerprints.main, ["--version"])
    assert result.exit_code == 0


def test_visualize_fingerprints_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_fingerprints.main, ["--help"])
    assert result.exit_code == 0
