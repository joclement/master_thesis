from pathlib import Path

import click.testing
import pytest
import yaml

from thesis import classify, data


@pytest.fixture
def config(classify_config):
    return classify_config


def test_classify_main(config, tmpdir):
    config["models-to-run"] = config["models-to-run"][0:1]
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(config, outfile)

    runner = click.testing.CliRunner()
    result = runner.invoke(classify.main, [str(config_filepath)])

    assert result.exit_code == 0


def test_classify_ClassificationHandler_with_saving(
    classify_config_with_tsfresh, tmpdir
):
    config = classify_config_with_tsfresh
    output_dir = Path(config["general"]["output_dir"])
    config["general"]["calc_cm"] = True
    config["general"]["save_models"] = True

    handler = classify.ClassificationHandler(config)
    assert Path(output_dir, "preprocessor.p").exists()
    assert Path(output_dir, "finger_preprocessor.p").exists()
    handler.run()

    num_of_models = len(config["models-to-run"])
    assert len(list(output_dir.rglob("confusion_matrix_*.svg"))) == num_of_models * (
        config["general"]["cv"] + 1
    )
    num_of_mlp_models = len([m for m in config["models-to-run"] if "mlp-" in m])
    assert len(list(output_dir.rglob("model-*.p"))) == num_of_models - num_of_mlp_models
    assert (
        2 * num_of_mlp_models
        <= len(list(output_dir.rglob("pipeline_step*.p")))
        <= 4 * num_of_mlp_models
    )


def test_classify_ClassificationHandler_step_duration(classify_config):
    classify_config["models-to-run"].remove("mlp-tsfresh")
    classify_config["general"]["step_duration"] = "20 seconds"
    handler = classify.ClassificationHandler(classify_config)
    handler.run()


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


def test_classify_ClassificationHandler_max_len(config):
    config["general"]["max_len"] = 23
    config["general"]["split"] = True
    config["models-to-run"] = ["knn_dtw-twod"]

    handler = classify.ClassificationHandler(config)
    handler.run()


def test_classify_ClassificationHandler_normalize_pd_values(config):
    config["models-to-run"] = ["rf-finger_ott"]
    config["general"]["normalize_fingerprints"] = True

    handler = classify.ClassificationHandler(config)
    for df in handler.measurements:
        assert df[data.PD].max() == 1.0


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
