import math

import pandas as pd

from thesis import fingerprint


def test_fingerprint_tu_graz(measurement):
    finger = fingerprint.tu_graz(measurement)
    assert type(finger) is pd.Series
    assert len(finger) == 14
    assert all([not math.isnan(value) for value in finger])
    assert finger.dtype == "float64"
    assert len(finger) == len(set(finger.tolist()))


def test_fingerprint_build_set(measurements):
    dataset = fingerprint.build_set(measurements)
    assert type(dataset) is pd.DataFrame
    assert dataset.shape == (5, 15)
