import numpy as np
import pandas as pd
import pytest

from thesis import data, prepared_data


@pytest.fixture
def simple_artificial_measurement():
    df = pd.DataFrame(data={data.TIME: [10.1, 23.2, 100.3], data.PD: [1, 2, 3]})
    return df


def test__convert_to_time_series(simple_artificial_measurement):
    time_series = prepared_data._convert_to_time_series(
        simple_artificial_measurement, "2ms"
    )
    assert len(time_series) == (100 - 10) / 2 + 1
    assert len(time_series == 0.0) == len(time_series)
    assert time_series[0] == 1
    assert time_series[6] == 2
    assert time_series[-1] == 3


def test_convert_to_tsfresh_dataset(measurements):
    dataset = prepared_data.convert_to_tsfresh_dataset(measurements)
    assert type(dataset) is pd.DataFrame


def test_seqfinger_ott(measurements):
    config = {"duration": "30 seconds"}
    X, scaler = prepared_data.seqfinger_ott(measurements, **config)
    X_transformed = scaler.fit_transform(X)
    assert X.shape == X_transformed.shape
    assert np.nanmax(X_transformed) == pytest.approx(1.0)
    assert np.nanmin(X_transformed) == 0.0
