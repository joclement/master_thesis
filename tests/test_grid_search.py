from pathlib import Path

import click.testing
import pytest
import yaml

from thesis import grid_search


@pytest.mark.expensive
def test_grid_search(classify_config_with_tsfresh, tmpdir):
    classify_config_with_tsfresh["general"]["cv"] = "group"
    classify_config_with_tsfresh["models-to-run"] = [
        "mlp-tsfresh",
        "dt-finger_own",
        "knn_dtw-oned",
    ]
    grid_searcher = grid_search.MyGridSearch(classify_config_with_tsfresh)
    assert len(grid_searcher.cv_splits) == 4
    grid_searcher.run()


@pytest.mark.expensive
def test_fingerprint_compare_grid(classify_config, tmpdir):
    classify_config["general"]["cv"] = "group"
    classify_config["models-to-run"] = [
        "mlp-finger_own",
    ]
    classify_config["models"]["mlp-finger_own"]["grid"] = "fingerprint_compare"

    grid_search.MyGridSearch(classify_config).run()


def test_grid_search_main(classify_config, tmpdir):
    classify_config["general"]["cv"] = "group"
    classify_config["models-to-run"] = ["dt-finger_ott"]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(classify_config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, [str(config_filepath)])

    assert result.exit_code == 0
    output_dir = Path(classify_config["general"]["output_dir"])
    num_of_models = len(classify_config["models-to-run"])
    assert len(list(output_dir.rglob("grid-search-results.p"))) == num_of_models


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, ["--help"])
    assert result.exit_code == 0
