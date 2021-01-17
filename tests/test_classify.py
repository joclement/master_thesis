from pathlib import Path

import click.testing
import pytest
import yaml

from thesis import classify, data
from thesis.tsfresh_features import save_extract_features


@pytest.fixture
def config(classify_config):
    return classify_config


@pytest.fixture
def config_with_tsfresh(config, multiple_csv_files):
    data_dir = Path(config["general"]["data_dir"])
    extracted_features_path = Path(data_dir, "extracted_features.data")
    save_extract_features(data.read_recursive(data_dir)[0], 1, extracted_features_path)
    for model_config in config["models"]:
        if "tsfresh_data" in config["models"][model_config]:
            config["models"][model_config]["tsfresh_data"] = str(
                extracted_features_path
            )
    return config


def test_classify_main(config, tmpdir):
    config["models-to-run"] = config["models-to-run"][0:2]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [str(config_filepath)])

    assert result.exit_code == 0
    assert Path(config["general"]["output_dir"], "train_val_scores.svg").exists()
    assert Path(config["general"]["output_dir"], "accuracy.svg").exists()
    assert Path(config["general"]["output_dir"], "top_3_accuracy.svg").exists()


def test_classify_ClassificationHandler(config_with_tsfresh, tmpdir):
    config = config_with_tsfresh
    output_dir = Path(config["general"]["output_dir"])
    config["general"]["calc_cm"] = True
    config["general"]["save_models"] = True

    num_of_models = len(config["models-to-run"])

    handler = classify.ClassificationHandler(config)
    handler.run()
    assert Path(config["general"]["output_dir"], "train_val_scores.svg").exists()
    assert Path(config["general"]["output_dir"], "accuracy.svg").exists()
    assert Path(config["general"]["output_dir"], "top_3_accuracy.svg").exists()

    assert len(list(output_dir.rglob("confusion_matrix_*.svg"))) == num_of_models * (
        config["general"]["cv"] + 1
    )
    num_of_mlp_models = len([m for m in config["models-to-run"] if "mlp-" in m])
    assert len(list(output_dir.rglob("model.p"))) == num_of_models - num_of_mlp_models


def test_classify_ClassificationHandler_no_defects(config):
    config["defects"] = list()

    with pytest.raises(ValueError):
        classify.ClassificationHandler(config)


def test_main_version_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--version"])
    assert result.exit_code == 0


def test_main_help_succeeds():
    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, ["--help"])
    assert result.exit_code == 0
