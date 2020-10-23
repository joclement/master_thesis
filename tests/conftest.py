from pathlib import Path

import pytest

from thesis import data


@pytest.fixture
def csv_folder():
    return "./testdata"


@pytest.fixture
def csv_filepath(csv_folder):
    return str(Path(csv_folder, "(+DC)_Partikel.csv"))


@pytest.fixture
def measurement(csv_filepath):
    return data.read(csv_filepath)


@pytest.fixture
def measurements(csv_folder):
    return data.read_recursive(csv_folder)[0]
