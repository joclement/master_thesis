import pandas as pd

from thesis import data


def test_data_read(csv_filepath):
    df = data.read(csv_filepath)
    assert data.TIME in df
    assert data.TIMEDIFF in df
    assert data.PD in df

    assert data.CLASS in df
    assert df[data.CLASS][0] == data.Defect.free_particle
    assert len(set(df[data.CLASS])) == 1

    assert data.VOLTAGE_SIGN in df
    assert df[data.VOLTAGE_SIGN][0] == data.VoltageSign.positive
    assert len(set(df[data.VOLTAGE_SIGN])) == 1


def test_data_read_recursive(csv_folder):
    measurements, csv_filepaths = data.read_recursive(csv_folder)
    assert len(csv_filepaths) == 2
    assert len(measurements) == 2
    assert type(measurements[0]) is pd.DataFrame


def test_data_normalize(csv_filepath):
    df = data.read(csv_filepath)
    data.Normalizer([df]).apply([df])
    assert df[data.PD].max() == 1.0
    assert df[data.TIMEDIFF].max() == 1.0
    assert df[data.TIMEDIFF].min() >= 0.0


def test_data_split_train_test(csv_folder):
    measurements, _ = data.read_recursive(csv_folder)
    dataset = 10 * [measurements[0]] + 10 * [measurements[1]]
    assert len(dataset) == 20
    train, test = data.split_train_test(dataset, test_size=0.3)
    assert len(train) == 14
    assert len(test) == 6
    classes_in_train = [df[data.CLASS][0] for df in train]
    classes_in_test = [df[data.CLASS][0] for df in test]
    assert (
        set(classes_in_train)
        == set(classes_in_test)
        == set([data.Defect.free_particle, data.Defect.particle_insulator])
    )
    assert classes_in_train.count(data.Defect.free_particle) == 7
    assert classes_in_test.count(data.Defect.particle_insulator) == 3


def test_data_Defect():
    assert data.Defect.free_particle == 0
    assert data.Defect.free_particle.name == "free_particle"
