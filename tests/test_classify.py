from itertools import product
from pathlib import Path
from shutil import copyfile

import click.testing
import pytest


from thesis import classify


@pytest.fixture
def multiple_csv_files(csv_folder, tmpdir):
    for idx, csv_file in product(range(5), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return str(tmpdir)


def test_classify_main_succeeds(multiple_csv_files):
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [multiple_csv_files])
    assert result.exit_code == 0
    assert "Accuracy for LukasMeanDist with fingerprint tu_graz: 1.0" in result.output
    assert "Accuracy for LukasMeanDist with fingerprint lukas: 1.0" in result.output

    assert (
        "Accuracy for KNeighborsClassifier with fingerprint tu_graz: 1.0"
        in result.output
    )
    assert (
        "Accuracy for KNeighborsClassifier with fingerprint lukas: 1.0" in result.output
    )

    assert "Accuracy for SVC with fingerprint tu_graz: 1.0" in result.output
    assert "Accuracy for SVC with fingerprint lukas: 1.0" in result.output

    # FIXME Accuracy only 0.2 for this classifier and fingerprint
    assert "Accuracy for MLPClassifier with fingerprint tu_graz:" in result.output
    assert "Accuracy for MLPClassifier with fingerprint lukas: 1.0" in result.output

    assert result.output.count("Confusion matrix") == 8


def test_classify_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_classify_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
