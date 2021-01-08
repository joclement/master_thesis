from itertools import product
from pathlib import Path
from shutil import copyfile

import click.testing
import pytest
import yaml

from thesis import classify


@pytest.fixture
def config(multiple_csv_files, tmpdir):
    with open("./config/test.yml", "r") as stream:
        config = yaml.safe_load(stream)
    config["general"]["data_dir"] = str(multiple_csv_files)
    config["general"]["output_dir"] = str(Path(tmpdir, "output"))
    return config


@pytest.fixture
def multiple_csv_files(csv_folder, tmpdir):
    for idx, csv_file in product(range(4), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return str(tmpdir)


def test_classify_main_succeeds(config, tmpdir):
    config["models-to-run"] = config["models-to-run"][0:2]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [str(config_filepath)])

    assert result.exit_code == 0
    assert Path(config["general"]["output_dir"], "models_all_bar.svg").exists()


def test_classify_main_calc_confusion_matrix_and_save_models_succeeds(config, tmpdir):
    output_dir = Path(config["general"]["output_dir"])
    config["general"]["calc_cm"] = True
    config["general"]["save_models"] = True

    num_of_models = len(config["models-to-run"])

    handler = classify.ClassificationHandler(config)
    handler.run()
    assert Path(output_dir, "models_all_bar.svg").exists()

    assert len(list(output_dir.rglob("confusion_matrix_*.svg"))) == num_of_models * (
        config["general"]["cv"] + 1
    )
    assert len(list(output_dir.rglob("model.p"))) == num_of_models


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
