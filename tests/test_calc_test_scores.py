from pathlib import Path

import click.testing
import pytest

from thesis import calc_test_scores, classify


@pytest.fixture
def results_dir_with_saved_models(classify_config):
    classify_config["models-to-run"] = classify_config["models-to-run"][0:1]
    classify_config["general"]["save_models"] = True
    classify.ClassificationHandler(classify_config).run()
    return classify_config["general"]["output_dir"]


def test_main(csv_folder, results_dir_with_saved_models, tmpdir):
    runner = click.testing.CliRunner()
    results_dir = Path(results_dir_with_saved_models)
    preprocessor_file = Path(results_dir, "preprocessor.p")
    model_files = list(results_dir.rglob("model-*.p"))
    assert len(model_files) == 1
    model_file = model_files[0]

    result = runner.invoke(
        calc_test_scores.click_command,
        [str(csv_folder), str(preprocessor_file), str(model_file)],
    )

    assert result.exit_code == 0


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(calc_test_scores.click_command, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(calc_test_scores.click_command, ["--help"])
    assert result.exit_code == 0
