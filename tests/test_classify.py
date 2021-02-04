from pathlib import Path

import click.testing
import pytest
import yaml

from thesis import classify


@pytest.fixture
def config(classify_config):
    return classify_config


def test_classify_main(config, tmpdir):
    config["models-to-run"] = config["models-to-run"][0:2]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [str(config_filepath)])

    assert result.exit_code == 0
    assert Path(config["general"]["output_dir"], "val_balanced_accuracy.svg").exists()
    assert Path(config["general"]["output_dir"], "val_accuracy.svg").exists()
    assert Path(config["general"]["output_dir"], "val_top_3_accuracy.svg").exists()


def test_classify_ClassificationHandler(config, tmpdir):
    output_dir = Path(config["general"]["output_dir"])
    config["general"]["calc_cm"] = True
    config["general"]["save_models"] = False

    handler = classify.ClassificationHandler(config)
    handler.run()

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
    assert Path(output_dir, "preprocessor.p").exists()

    num_of_models = len(config["models-to-run"])
    num_of_mlp_models = len([m for m in config["models-to-run"] if "mlp-" in m])
    assert len(list(output_dir.rglob("model-*.p"))) == num_of_models - num_of_mlp_models
    assert len(list(output_dir.rglob("pipeline_step*.p"))) == num_of_mlp_models * 2


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
