from pathlib import Path

import pytest


@pytest.fixture
def csv_folder():
    return "./testdata"


@pytest.fixture
def csv_filepath(csv_folder):
    return str(Path(csv_folder, "(+DC)_Partikel.csv"))
