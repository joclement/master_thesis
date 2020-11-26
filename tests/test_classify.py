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
    assert result.output.count("Confusion matrix") == 0

    assert Path(tmpdir, "classifiers_balanced_accuracy_bar.svg").exists()
    assert not Path(tmpdir, "confusion_matrix_SVM_fingerprint_Ott.svg").exists()
    assert len(list(Path(tmpdir).rglob("confusion_matrix_*.svg"))) == 0


def test_classify_main_calc_confusion_matrix_succeeds(multiple_csv_files, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        classify.main, ["--calc-cm", multiple_csv_files, str(tmpdir)]
    )
    assert result.exit_code == 0

    ones = np.ones(4)
    assert f"Scores for Ott with fingerprint TU Graz: {ones}" in result.output
    assert f"Scores for Ott with fingerprint Ott: {ones}" in result.output
    assert f"Scores for Ott with fingerprint Ott + TU Graz: {ones}" in result.output

    assert f"Scores for 1-NN with fingerprint TU Graz: {ones}" in result.output
    assert f"Scores for 3-NN with fingerprint Ott: {ones}" in result.output
    assert f"Scores for 3-NN with fingerprint Ott + TU Graz: {ones}" in result.output

    assert f"Scores for SVM with fingerprint TU Graz: {ones}" in result.output
    assert f"Scores for SVM with fingerprint Ott: {ones}" in result.output
    assert f"Scores for SVM with fingerprint Ott + TU Graz: {ones}" in result.output

    assert "Scores for MLP with fingerprint TU Graz: " in result.output
    assert "Scores for MLP with fingerprint Ott:" in result.output
    assert "Scores for MLP with fingerprint Ott + TU Graz: " in result.output

    assert "Scores for 1-NN 1D DTW with Time Series: " in result.output
    assert "Scores for 3-NN 1D DTW with Time Series: " in result.output

    assert "Scores for 1-NN 2D DTW with Time Series: " in result.output
    assert "Scores for 3-NN 2D DTW with Time Series: " in result.output

    assert result.output.count("Confusion matrix") == 19

    assert len(list(Path(tmpdir).rglob("confusion_matrix_*.svg"))) == 19
    assert Path(tmpdir, "confusion_matrix_SVM_fingerprint_Ott.svg").exists()


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
