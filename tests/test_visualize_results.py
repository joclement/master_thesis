import click.testing
import pytest

from jc_thesis_code import classify, visualize_results


@pytest.fixture
def results_dir(classify_config):
    classify_config["models-to-run"] = classify_config["models-to-run"][0:2]
    classify.ClassificationHandler(classify_config).run()
    return classify_config["general"]["output_dir"]


def test_main(results_dir, tmpdir):
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_results.main, [results_dir, "--no-show"])

    assert result.exit_code == 0


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_results.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(visualize_results.main, ["--help"])
    assert result.exit_code == 0
