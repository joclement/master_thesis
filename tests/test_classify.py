from pathlib import Path

import click.testing
import pandas as pd
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
    assert len(list(output_dir.rglob("model-*.p"))) == num_of_models
    num_of_mlp_models = len([m for m in config["models-to-run"] if "mlp-" in m])
    assert len(list(output_dir.rglob("keras.h5"))) == num_of_mlp_models


def test_classify_ClassificationHandler_step_duration(classify_config):
    classify_config["models-to-run"].remove("mlp-tsfresh")
    classify_config["models-to-run"].remove("lr-oned_weasel")
    classify_config["general"]["step_duration"] = "20 seconds"
    handler = classify.ClassificationHandler(classify_config)
    handler.run()


def test_classify_ClassificationHandler_step_duration_no_balance_weasel(
    classify_config,
):
    classify_config["models-to-run"] = ["lr-oned_weasel"]
    classify_config["general"]["balance"] = False
    classify_config["general"]["step_duration"] = "30 seconds"
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
    config["general"]["min_duration"] = None
    config["general"]["max_duration"] = None
    config["general"]["max_len"] = 23
    config["general"]["repeat"] = 2
    config["general"]["split"] = True
    config["models-to-run"] = ["knn_dtw-twod"]

    handler = classify.ClassificationHandler(config)
    handler.run()


def different_polarity(df):
    new_df = df.copy()
    new_df.attrs[data.VOLTAGE_SIGN] = not new_df.attrs[data.VOLTAGE_SIGN]
    return new_df


def test_classify_compute_sample_weight(measurements):
    polarity_measurements = []
    for df in measurements:
        polarity_measurements.append(different_polarity(df))
        polarity_measurements.append(different_polarity(df))
    polarity_measurements.extend(measurements)
    y = pd.Series(
        data=data.get_defects(polarity_measurements),
        index=classify.build_index(polarity_measurements),
    )

    sample_weights = classify.compute_sample_weight(polarity_measurements, y, True)
    assert len(sample_weights) == len(polarity_measurements)
    assert len(set(sample_weights[: 2 * len(measurements)])) == 1
    assert len(set(sample_weights[2 * len(measurements) :])) == 1
    assert sample_weights[0] * 2 == sample_weights[2 * len(measurements)]


def test_classify_ClassificationHandler_normalize_pd_values(config):
    config["general"]["normalize_pd"] = True

    handler = classify.ClassificationHandler(config)
    assert all([df[data.PD].max() <= 1.0 for df in handler.measurements])


def test_classify_ClassificationHandler_normalize_pd_values_no_split(config):
    config["general"]["normalize_pd"] = True
    config["general"]["split"] = False

    handler = classify.ClassificationHandler(config)
    assert all([df[data.PD].max() == 1.0 for df in handler.measurements])


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
