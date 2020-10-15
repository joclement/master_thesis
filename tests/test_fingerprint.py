import pandas as pd
import pytest

from thesis import data, fingerprint


@pytest.fixture
def measurement(csv_filepath):
    return data.read(csv_filepath)


def test_fingerprint_tu_graz(measurement):
    finger = fingerprint.tu_graz(measurement)
    assert type(finger) is pd.Series
    assert len(finger) == 14
    assert len(finger) == len(set(finger.tolist()))
