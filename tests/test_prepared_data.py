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


def test__convert_to_time_series_same_frequency(simple_artificial_measurement):
    time_series = prepared_data._convert_to_time_series(
        simple_artificial_measurement, "50us"
    )
    assert time_series.equals(time_series.resample("50us").max())


def test_split_by_durations(measurement):
    assert "path" in measurement.attrs
    splitted = prepared_data.split_by_durations(
        [measurement], pd.Timedelta("30 seconds")
    )

    assert all([df.attrs["path"] == measurement.attrs["path"] for df in splitted])
    assert len(splitted) == 4
    assert [df.attrs["part"] for df in splitted] == list(range(len(splitted)))
    assert all([isinstance(df, pd.DataFrame) for df in splitted])
    assert splitted[2][data.PD].iloc[-1] == 3
    assert len(splitted[-1]) == 9
    assert splitted[-1][data.PD].iloc[0] == 67
    assert all([all(df.index == range(len(df.index))) for df in splitted])

    splitted = prepared_data.split_by_durations(
        [measurement], pd.Timedelta("60 seconds")
    )
    assert len(splitted) == 2


def test__split_by_duration(measurement):
    splitted = prepared_data._split_by_duration(
        measurement, pd.Timedelta("10 seconds"), True
    )

    assert isinstance(splitted, list)
    assert len(splitted) == 12
    assert len(splitted[0]) == 14
    assert splitted[0][data.TIME].iloc[-1] == 10000
    assert splitted[6][data.TIME].iloc[1] == 62000

    splitted = prepared_data._split_by_duration(
        measurement, pd.Timedelta("10 seconds"), False
    )
    assert len(splitted) == 13


def test_reset_times_with_normal_measurements(measurements):
    assert all(
        [df[data.TIME].iloc[0] == df[data.TIME_DIFF].iloc[0] for df in measurements]
    )
    reseted = prepared_data.reset_times(measurements)
    assert all([df[data.TIME].iloc[0] == df[data.TIME_DIFF].iloc[0] for df in reseted])
    reseted = prepared_data.reset_times(measurements)
    assert all([df[data.TIME].iloc[0] == df[data.TIME_DIFF].iloc[0] for df in reseted])


def test_reset_times_with_splitted_measurements(measurements):
    splitted = prepared_data.split_by_durations(
        measurements, pd.Timedelta("20 seconds")
    )
    assert not all(
        [df[data.TIME].iloc[0] == df[data.TIME_DIFF].iloc[0] for df in splitted]
    )

    reseted = prepared_data.reset_times(splitted)
    assert all([df[data.TIME].iloc[0] == df[data.TIME_DIFF].iloc[0] for df in reseted])
