import pandas as pd

from thesis import tsfresh_features


def test_main(csv_folder):
    relevance_table = tsfresh_features.main(csv_folder)
    assert type(relevance_table) is pd.DataFrame


def test_convert_to_tsfresh_dataset(measurements):
    dataset = tsfresh_features.convert_to_tsfresh_dataset(measurements)
    assert type(dataset) is pd.DataFrame
