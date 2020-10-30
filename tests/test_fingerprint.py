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

    assert finger[fingerprint.PDS_PER_SEC] > 0
    assert math.isclose(finger[fingerprint.PDS_PER_SEC], 2848, rel_tol=0.05)


def test_fingerprint_build_set(measurements):
    dataset = fingerprint.build_set(measurements)
    assert type(dataset) is pd.DataFrame
    assert dataset.shape == (5, 15)


def test_fingerprint_lukas(large_df):
    finger = fingerprint.lukas(large_df)
    assert type(finger) is pd.Series
    assert len(finger) == 12
    assert all([not math.isnan(value) for value in finger])
    assert finger.dtype == "float64"
    assert len(finger) == len(set(finger.tolist()))

    assert -1 <= finger[fingerprint.CORR_PD_DIFF_TO_PD] <= 1
    assert -1 <= finger[fingerprint.CORR_NEXT_PD_TO_PD] <= 1


def test_fingerprint_build_set_lukas(large_df):
    dataset = fingerprint.build_set([large_df], fingerprint=fingerprint.lukas)
    assert type(dataset) is pd.DataFrame
    assert dataset.shape == (1, 13)
