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
