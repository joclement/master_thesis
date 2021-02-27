import pandas as pd

from thesis import data, prepared_data


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
        measurement, pd.Timedelta("10 seconds"), True, drop_empty=True
    )

    assert isinstance(splitted, list)
    assert len(splitted) == 12
    assert len(splitted[0]) == 14
    assert len(splitted[-1]) == 4
    for df in splitted:
        print("\n\n")
        print(df)
    assert all([df[data.TIME_DIFF][1:].sum() <= 10000 for df in splitted])
    assert splitted[0][data.TIME_DIFF].iloc[-1] == 1000
    assert splitted[6][data.TIME_DIFF].iloc[0] == 1000

    splitted = prepared_data._split_by_duration(
        measurement, pd.Timedelta("10 seconds"), True, drop_empty=False
    )
    assert splitted[6][data.TIME_DIFF].iloc[0] == 3000

    splitted = prepared_data._split_by_duration(
        measurement, pd.Timedelta("10 seconds"), False
    )
    assert len(splitted) == 13
    assert len(splitted[0]) == 14
    assert len(splitted[-1]) == 2


def test_oned(measurements):
    config = {"fix_duration": "200 seconds", "frequency": "500ms"}
    dataset = prepared_data.oned_func(measurements, **config)

    assert len(dataset.shape) == 3
    # FIXME assert not np.isnan(dataset).any()
