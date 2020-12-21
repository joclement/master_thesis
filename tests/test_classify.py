from itertools import product
from pathlib import Path
from shutil import copyfile

import click.testing
import pytest

from thesis import classify


@pytest.fixture
def multiple_csv_files(csv_folder, tmpdir):
    for idx, csv_file in product(range(4), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return str(tmpdir)


def test_classify_main_succeeds(multiple_csv_files, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [multiple_csv_files, str(tmpdir)])
    assert result.exit_code == 0
    assert result.output.count("Confusion matrix") == 0

    assert Path(tmpdir, "classifiers_balanced_accuracy_bar.svg").exists()
    assert len(list(Path(tmpdir).rglob("confusion_matrix_*.svg"))) == 0


@pytest.mark.expensive
def test_classify_main_calc_confusion_matrix_succeeds(multiple_csv_files, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(
        classify.main, ["--calc-cm", multiple_csv_files, str(tmpdir)]
    )
    assert result.exit_code == 0

    assert (
        result.output.count("Confusion matrix")
        == result.output.count("Scores for")
        == len(list(Path(tmpdir).rglob("confusion_matrix_*.svg")))
    )


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
