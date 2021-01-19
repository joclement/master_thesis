import pandas as pd

from thesis import data, util


def test_todataTime():
    assert data.TIME_UNIT == "ms"

    duration = pd.Timedelta("30 seconds")
    assert util.to_dataTIME(duration) == 30000
    assert type(util.to_dataTIME(duration)) is int
