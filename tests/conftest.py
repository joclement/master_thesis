from pathlib import Path

import pytest

from thesis import data


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "expensive: test takes too much time to run normally with nox."
    )
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")


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


@pytest.fixture
def tiny_csv_folder(testdata):
    return str(Path(testdata, "small_visualize"))


@pytest.fixture
def tiny_csv_filepath(tiny_csv_folder):
    return str(Path(tiny_csv_folder, "(+DC) Spitze an HS_FOO.csv"))
