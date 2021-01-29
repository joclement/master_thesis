from pathlib import Path

import click.testing
import pandas as pd
import pytest
import yaml

from thesis import classify, data
from thesis.prepared_data import split_by_durations
from thesis.tsfresh_features import save_extract_features


@pytest.fixture
def config(classify_config):
    return classify_config


@pytest.fixture
def config_with_tsfresh(config, multiple_csv_files):
    data_dir = Path(config["general"]["data_dir"])
    extracted_features_path = Path(data_dir, "extracted_features.data")
    splitted = split_by_durations(
        data.read_recursive(data_dir)[0],
        pd.Timedelta(config["general"]["max_duration"]),
    )
    save_extract_features(splitted, 1, extracted_features_path, True)
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
    config["general"]["save_models"] = False

    handler = classify.ClassificationHandler(config)
    handler.run()
    assert Path(config["general"]["output_dir"], "train_val_scores.svg").exists()
    assert Path(config["general"]["output_dir"], "accuracy.svg").exists()
    assert Path(config["general"]["output_dir"], "top_3_accuracy.svg").exists()

    num_of_models = len(config["models-to-run"])
    assert len(list(output_dir.rglob("confusion_matrix_*.svg"))) == num_of_models * (
        config["general"]["cv"] + 1
    )


def test_classify_ClassificationHandler_save_models(config):
    config["models-to-run"] = ["mlp-finger_own", "dt-finger_ott"]
    config["general"]["save_models"] = True

    handler = classify.ClassificationHandler(config)
    handler.run()

    output_dir = Path(config["general"]["output_dir"])
    num_of_models = len(config["models-to-run"])
    num_of_mlp_models = len([m for m in config["models-to-run"] if "mlp-" in m])
    assert len(list(output_dir.rglob("model.p"))) == num_of_models - num_of_mlp_models
    assert len(list(output_dir.rglob("pipeline_step*.p"))) == num_of_mlp_models


def test_classify_ClassificationHandler_no_defects(config):
    config["defects"] = list()

    with pytest.raises(ValueError):
        classify.ClassificationHandler(config)


def test_classify_ClassificationHandler_logo_cv(config):
    config["general"]["cv"] = "logo"
    config["models-to-run"] = config["models-to-run"][0:1]
    config["general"]["calc_cm"] = True

    handler = classify.ClassificationHandler(config)
    handler.run()
    output_dir = Path(config["general"]["output_dir"])
    assert len(list(output_dir.rglob("confusion_matrix_*.svg"))) == 1


def test_classify_ClassificationHandler_invalid_cv(config):
    config["general"]["cv"] = "invalid"

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
