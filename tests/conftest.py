from pathlib import Path

import pytest

from thesis import data


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "expensive: mark as expensive test, should not run by default."
    )


@pytest.fixture
def testdata():
    return "./testdata"


@pytest.fixture
def csv_folder(testdata):
    return str(Path(testdata, "small"))


@pytest.fixture
def csv_filepath(csv_folder):
    return str(Path(csv_folder, "(+DC)_Partikel.csv"))


@pytest.fixture
def measurement(csv_filepath):
    return data.read(csv_filepath)


@pytest.fixture
def measurements(csv_folder):
    return data.read_recursive(csv_folder)[0]


@pytest.fixture
def large_csv_filepath(testdata):
    return str(Path(testdata, "large", "(+DC)_Partikel_large.csv"))


@pytest.fixture
def large_df(large_csv_filepath):
    return data.read(large_csv_filepath)
