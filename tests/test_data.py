from pathlib import Path

import pandas as pd

from thesis import data


def test_data_read(csv_filepath):
    df = data.read(csv_filepath)
    assert data.CLASS in df
    assert df[data.CLASS][0] == data.Defect.free_particle
    assert len(set(df[data.CLASS])) == 1
    assert data.VOLTAGE_SIGN in df
    assert df[data.VOLTAGE_SIGN][0] == data.VoltageSign.positive
    assert len(set(df[data.VOLTAGE_SIGN])) == 1


def test_data_read_recursive(csv_filepath):
    measurements, csv_filepaths = data.read_recursive(Path(csv_filepath).parent)
    assert len(csv_filepaths) == 1
    assert csv_filepaths[0] == Path(csv_filepath)
    assert len(measurements) == 1
    assert type(measurements[0]) is pd.DataFrame
