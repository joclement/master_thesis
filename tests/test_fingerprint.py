import math
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import approx, fixture, mark
from sklearn.preprocessing import MinMaxScaler

from thesis import data, fingerprint


def test_fingerprint_tu_graz(measurement):
    finger = fingerprint.tu_graz(measurement)
    assert type(finger) is pd.Series
    assert len(finger) == 14
    assert all([not math.isnan(value) for value in finger])
    assert finger.dtype == "float64"
    assert len(finger) == len(set(finger.tolist()))

    assert finger[fingerprint.PD_DIFF_WEIB_A] >= 0
    assert finger[fingerprint.PD_DIFF_WEIB_B] >= 0

    assert finger[fingerprint.PDS_PER_SEC] > 0


def test_fingerprint_lukas(large_df):
    finger = fingerprint.lukas(large_df)
    assert type(finger) is pd.Series
    assert len(finger) == 12
    assert all([not math.isnan(value) for value in finger])
    assert finger.dtype == "float64"
    assert len(finger) == len(set(finger.tolist()))

    assert finger[fingerprint.PD_DIFF_MEAN] > 0
    assert finger[fingerprint.PD_DIFF_WEIB_A] >= 0

    assert -1 <= finger[fingerprint.CORR_PD_DIFF_TO_PD] <= 1
    assert -1 <= finger[fingerprint.CORR_NEXT_PD_TO_PD] <= 1


def test_fingerprint_build_set_lukas(large_df):
    dataset = fingerprint.build_set([large_df], fingerprint=fingerprint.lukas)
    assert type(dataset) is pd.DataFrame
    assert dataset.shape == (1, 12)


def test_normalize_fingerprints(measurements):
    fingerprints = fingerprint.build_set(measurements, fingerprint.own)
    scaler = MinMaxScaler()
    scaler.fit(fingerprints)
    normalized_fingers = scaler.transform(fingerprints)

    for max_value in np.amax(normalized_fingers, axis=0):
        assert math.isclose(max_value, 1.0) or max_value == 0.0
    assert all(np.amin(normalized_fingers, axis=0) == 0)


def test_fingerprint_own(measurements):
    dataset = fingerprint.build_set(measurements, fingerprint=fingerprint.own)
    assert dataset.shape == (6, 24)
    assert fingerprint.POLARITY not in dataset


@fixture
def real_data_folder() -> Path:
    folder = Path(
        "./data/csv-files/prepared/older_mat_converted/unique/3_entities_or_more",
    )
    assert folder.is_dir()
    return folder


@mark.expensive
@mark.e2e
def test_lukas_fingerprint_results_with_master_thesis_values(real_data_folder):
    df = data.read(
        Path(real_data_folder, "(-DC) Partikel 6mm d=0,5mm 0,6 MPa 1,2Uab.csv")
    )

    pd = df[data.PD]
    assert pd.median() == approx(7.619, abs=1e-3)
    assert pd.mode()[0] == approx(7.710, abs=1e-3)
    assert pd.min() == approx(0.500, abs=1e-3)
    assert pd.max() == approx(9.914, abs=1e-3)
    assert pd.std() / pd.mean() == approx(0.154, abs=1e-3)

    pd_diff = df[data.PD_DIFF]
    assert pd_diff.mean() == approx(0.388, abs=1e-3)
    assert pd_diff.skew() == approx(5.381, abs=1e-3)
    # @note: the Matlab implementation of kurtosis done by Lukas is biased,
    #        so slightly different from pandas.
    assert pd_diff.kurt() == approx(50.985, rel=1e-1)

    finger = fingerprint.lukas(df)

    assert finger[fingerprint.PD_MEAN] == approx(7.457, abs=1e-3)
    assert finger[fingerprint.PD_MAX] == approx(9.914, abs=1e-3)
    assert finger[fingerprint.PD_CV] == approx(0.154, abs=1e-3)

    assert finger[fingerprint.PD_DIFF_MEAN] == approx(0.388, abs=1e-3)
    assert finger[fingerprint.PD_DIFF_SKEW] == approx(5.381, abs=1e-3)
    # @note: the Matlab implementation of kurtosis done by Lukas is biased,
    #        so slightly different from pandas.
    assert finger[fingerprint.PD_DIFF_KURT] == approx(50.985, rel=1e-1)

    assert finger[fingerprint.TD_MEDIAN] == approx(0.091, abs=1e-3)

    # FIXME Issue #22: Weibull parameters are not the same as in Lukas thesis
    # assert finger[fingerprint.PD_WEIB_A] == approx(7.917, abs=1e-3)
    # assert finger[fingerprint.PD_WEIB_B] == approx(8.277, abs=1e-3)
    # assert finger[fingerprint.PD_DIFF_WEIB_A] == approx(0.391, abs=1e-3)

    assert finger[fingerprint.CORR_PD_DIFF_TO_PD] == approx(-0.908, abs=1e-1)
    assert finger[fingerprint.CORR_NEXT_PD_TO_PD] == approx(0.503, abs=1e-2)
