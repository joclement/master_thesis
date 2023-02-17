from pathlib import Path

import click.testing
import pytest

from jc_thesis_code import calc_test_scores, classify


@pytest.fixture
def results_dir_with_saved_models(classify_config):
    classify_config["models-to-run"] = classify_config["models-to-run"][0:1]
    classify_config["general"]["save_models"] = True
    classify.ClassificationHandler(classify_config).run()
    return classify_config["general"]["output_dir"]


def test_click_command(csv_folder, results_dir_with_saved_models, tmpdir):
    runner = click.testing.CliRunner()
    results_dir = Path(results_dir_with_saved_models)
    preprocessor_file = Path(results_dir, "preprocessor.p")
    finger_preprocessor_file = Path(results_dir, "finger_preprocessor.p")
    model_files = list(results_dir.rglob("model-*.p"))
    assert len(model_files) == 1
    model_file = model_files[0]

    test_scores = Path(tmpdir, "test_scores.csv")
    assert not test_scores.exists()
    result = runner.invoke(
        calc_test_scores.click_command,
        [
            "-t",
            str(csv_folder),
            "-p",
            str(preprocessor_file),
            "-m",
            str(model_file),
            "-f",
            str(finger_preprocessor_file),
            "-o",
            str(test_scores),
        ],
    )

    assert result.exit_code == 0
    assert test_scores.exists()


def test_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(calc_test_scores.click_command, ["--version"])
    assert result.exit_code == 0


def test_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(calc_test_scores.click_command, ["--help"])
    assert result.exit_code == 0
