from pathlib import Path

import click.testing
import yaml

from thesis import grid_search


def test_grid_search(classify_config, tmpdir):
    classify_config["models-to-run"] = ["mlp-finger_own", "dt-finger_ott"]
    grid_search.MyGridSearch(classify_config).run()


def test_grid_search_main(classify_config, tmpdir):
    classify_config["models-to-run"] = ["dt-finger_ott"]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(classify_config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, [str(config_filepath)])

    assert result.exit_code == 0


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(grid_search.main, ["--help"])
    assert result.exit_code == 0