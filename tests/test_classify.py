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
    np.random.seed(2)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [multiple_csv_files, str(tmpdir)])
    assert result.exit_code == 0
    ones = np.ones(4)
    assert f"Accuracies for Ott with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for Ott with fingerprint Ott: {ones}" in result.output
    assert f"Accuracies for Ott with fingerprint Ott + TU Graz: {ones}" in result.output

    assert f"Accuracies for 1-NN with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for 3-NN with fingerprint Ott: {ones}" in result.output
    assert (
        f"Accuracies for 3-NN with fingerprint Ott + TU Graz: {ones}" in result.output
    )

    assert f"Accuracies for SVM with fingerprint TU Graz: {ones}" in result.output
    assert f"Accuracies for SVM with fingerprint Ott: {ones}" in result.output
    assert f"Accuracies for SVM with fingerprint Ott + TU Graz: {ones}" in result.output

    assert "Accuracies for MLP with fingerprint TU Graz: " in result.output
    assert "Accuracies for MLP with fingerprint Ott:" in result.output
    assert "Accuracies for MLP with fingerprint Ott + TU Graz: " in result.output

    assert "Accuracies for 1-NN DTW with Time Series: " in result.output
    assert "Accuracies for 3-NN DTW with Time Series: " in result.output

    assert result.output.count("Confusion matrix") == 17

    assert Path(tmpdir, "classifiers_accuracy_bar.svg").exists()
    assert Path(tmpdir, "confusion_matrix_SVM_fingerprint_Ott.svg").exists()
    assert len(list(Path(tmpdir).rglob("confusion_matrix_*.svg"))) == 17


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
