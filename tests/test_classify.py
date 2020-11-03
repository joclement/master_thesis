from itertools import product
from pathlib import Path
from shutil import copyfile

import click.testing
import numpy as np
import pytest


from thesis import classify


@pytest.fixture
def multiple_csv_files(csv_folder, tmpdir):
    for idx, csv_file in product(range(4), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return str(tmpdir)


def test_classify_main_succeeds(multiple_csv_files, tmpdir):
    np.random.seed(11)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [multiple_csv_files, str(tmpdir)])
    assert result.exit_code == 0
    ones = np.ones(4)
    assert (
        f"Accuracies for LukasMeanDist with fingerprint tu_graz: {ones}"
        in result.output
    )
    assert (
        f"Accuracies for LukasMeanDist with fingerprint lukas: {ones}" in result.output
    )

    assert (
        f"Accuracies for KNeighborsClassifier with fingerprint tu_graz: {ones}"
        in result.output
    )
    assert (
        f"Accuracies for KNeighborsClassifier with fingerprint lukas: {ones}"
        in result.output
    )

    assert f"Accuracies for SVC with fingerprint tu_graz: {ones}" in result.output
    assert f"Accuracies for SVC with fingerprint lukas: {ones}" in result.output

    assert (
        f"Accuracies for MLPClassifier with fingerprint tu_graz: {ones}"
        in result.output
    )
    assert (
        f"Accuracies for MLPClassifier with fingerprint lukas: {ones}" in result.output
    )

    assert "lukas_plus_tu_graz" in result.output
    assert result.output.count("Confusion matrix") == 12

    assert Path(tmpdir, "bar.png").exists()


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
