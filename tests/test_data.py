from pathlib import Path
from shutil import copyfile

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from thesis import data


def test_data_read(csv_filepath):
    df = data.read(csv_filepath)
    assert data.TIME in df
    assert data.TIME_DIFF in df
    assert data.PD in df

    assert data.CLASS in df
    assert df[data.CLASS][0] == data.Defect.free_particle
    assert len(set(df[data.CLASS])) == 1

    assert data.VOLTAGE_SIGN in df.attrs
    assert df.attrs[data.VOLTAGE_SIGN] == data.VoltageSign.positive

    assert data.TEST_VOLTAGE not in df
    assert data.PD_DIFF in df
    assert len(df.columns) == 5


def test_data_read_wrong_filename(csv_filepath, tmpdir):
    wrong_file = Path(tmpdir, "wrong_filename.csv")
    copyfile(csv_filepath, wrong_file)
    with pytest.raises(ValueError):
        data.read(wrong_file)


def test_data_read_recursive(csv_folder):
    measurements, csv_filepaths = data.read_recursive(csv_folder)
    assert len(csv_filepaths) == len(measurements) == 6
    assert all([type(measurement) is pd.DataFrame for measurement in measurements])
    assert all(
        [
            len(measurement.columns) <= 7
            if data.TEST_VOLTAGE in measurement
            else len(measurement.columns) <= 6
            for measurement in measurements
        ]
    )


def test_data_clip_neg_values(measurement):
    measurement.loc[0, data.PD] = -0.01
    assert measurement[data.PD].min() == -0.01
    data.clip_neg_pd_values([measurement])
    assert measurement[data.PD].min() == 0.0


def test_data_split_train_test(measurements):
    dataset = 2 * measurements
    train, test = train_test_split(
        dataset, test_size=0.5, stratify=data.get_defects(dataset)
    )
    assert len(train) == len(dataset) / 2
    assert len(test) == len(dataset) / 2
    classes_in_train = [df[data.CLASS][0] for df in train]
    classes_in_test = [df[data.CLASS][0] for df in test]
    assert set(classes_in_train) == set(classes_in_test) == set(data.Defect)
    assert all(
        [classes_in_train.count(d) == classes_in_test.count(d) for d in data.Defect]
    )


def test_data_Defect():
    assert data.Defect.free_particle == 0
    assert data.Defect.free_particle.name == "free_particle"


def test_data_drop_columns_repeatively(csv_filepath):
    df = data.read(csv_filepath)
    assert data.TIME in df
    df.drop(columns=data.TIME, inplace=True)
    with pytest.raises(KeyError):
        df.drop(columns=data.TIME, inplace=True)
    df.drop(columns=data.TIME, inplace=True, errors="ignore")
