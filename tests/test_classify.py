from itertools import product
from pathlib import Path
from shutil import copyfile

import pytest
import yaml

from thesis import classify


@pytest.fixture
def config():
    with open("./config/test.yml", "r") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def multiple_csv_files(csv_folder, tmpdir):
    for idx, csv_file in product(range(4), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return str(tmpdir)


@pytest.fixture
def config_and_multiple_csv_files(config, csv_folder, tmpdir):
    config["general"]["data_dir"] = str(tmpdir)
    config["general"]["output_dir"] = str(Path(tmpdir, "output"))
    config_filepath = Path(tmpdir, "config.yml")
    with open(config_filepath, "w") as outfile:
        yaml.dump(config, outfile)
    for idx, csv_file in product(range(4), Path(csv_folder).glob("*.csv")):
        copyfile(csv_file, Path(tmpdir, f"{csv_file.stem}{idx}.csv"))
    return config_filepath, tmpdir


def test_classify_main_succeeds(config_and_multiple_csv_files, tmpdir):
    config_filepath, csv_folder = config_and_multiple_csv_files
    classify.main(config_filepath)

    assert Path(tmpdir, "output", "models_all_bar.svg").exists()


@pytest.mark.expensive
def test_classify_main_calc_confusion_matrix_succeeds(
    config, multiple_csv_files, tmpdir
):
    config["general"]["data_dir"] = str(multiple_csv_files)
    config["general"]["output_dir"] = str(Path(tmpdir, "output"))
    config["general"]["calc_cm"] = True

    handler = classify.ClassificationHandler(config)
    handler.run()
    assert len(list(Path(tmpdir).rglob("confusion_matrix_*.svg"))) > 0
