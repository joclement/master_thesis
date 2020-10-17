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


def test_data_normalize(csv_filepath):
    measurements, _ = data.read_recursive(Path(csv_filepath).parent)
    data.Normalizer(measurements).apply(measurements)
    assert measurements[0][data.PD].max() == 1.0
    assert measurements[0][data.TIMEDIFF].max() == 1.0
    assert measurements[0][data.TIMEDIFF].min() >= 0.0


def test_data_split_train_test():
    dataset = range(23)
    train, test = data.split_train_test(dataset)
    assert train == range(16)
    assert test == range(16, 23)


def test_data_Defect():
    assert data.Defect.free_particle == 0
    assert data.Defect.free_particle.name == "free_particle"
