import pandas as pd
from pytest import approx

from jc_thesis_code import data, util


def test_to_dataTime(measurement):
    assert data.TIME_UNIT == "ms"

    duration = pd.Timedelta("30 seconds")
    assert util.to_dataTIME(duration) == 30000
    assert type(util.to_dataTIME(duration)) is int

    assert measurement[data.TIME_DIFF].sum() == approx(129200)
