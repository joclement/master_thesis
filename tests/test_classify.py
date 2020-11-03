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
    np.random.seed(4)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [multiple_csv_files, str(tmpdir)])
    assert result.exit_code == 0
    ones = np.ones(4)
    assert (
        f"Accuracies for LukasMeanDist with fingerprint TU Graz: {ones}"
        in result.output
    )
    assert (
        f"Accuracies for LukasMeanDist with fingerprint Lukas: {ones}" in result.output
    )
    assert (
        f"Accuracies for LukasMeanDist with fingerprint Lukas + TU Graz: {ones}"
        in result.output
    )

    assert f"Accuracies for 1-NN with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for 3-NN with fingerprint Lukas: {ones}" in result.output
    assert (
        f"Accuracies for 3-NN with fingerprint Lukas + TU Graz: {ones}" in result.output
    )

    assert f"Accuracies for SVM with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for SVM with fingerprint Lukas: {ones}" in result.output
    assert (
        f"Accuracies for SVM with fingerprint Lukas + TU Graz: {ones}" in result.output
    )

    assert f"Accuracies for MLP with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for MLP with fingerprint Lukas: {ones}" in result.output
    assert (
        f"Accuracies for MLP with fingerprint Lukas + TU Graz: {ones}" in result.output
    )

    assert result.output.count("Confusion matrix") == 15

    assert Path(tmpdir, "bar.svg").exists()


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
