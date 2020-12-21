from pathlib import Path
import sys
from typing import Union

import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from . import data


def _drop_unneded_columns(measurements):
    for measurement in measurements:
        measurement.drop(
            columns=[data.TIME_DIFF, data.PD_DIFF, data.VOLTAGE_SIGN, data.CLASS],
            errors="raise",
            inplace=True,
        )
        measurement.drop(columns=[data.TEST_VOLTAGE], errors="ignore", inplace=True)


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")
    df[data.TIME] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
    df.set_index(data.TIME, inplace=True)
    time_series = df.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()
    time_series.reset_index(inplace=True)
    return time_series


def main(input_directory: Union[str, Path] = sys.argv[1]):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    """
    measurements, _ = data.read_recursive(input_directory)
    data.clip_neg_pd_values(measurements)
    y = pd.Series(data.get_defects(measurements))
    _drop_unneded_columns(measurements)

    frequency = pd.tseries.frequencies.to_offset("500ms")
    measurements = [_convert_to_time_series(df, frequency) for df in measurements]
    for index, df in enumerate(measurements):
        df["ID"] = index
    min_len = min([len(m) for m in measurements])
    measurements = [df[: min(len(df.index), 8 * min_len)] for df in measurements]
    all_df = pd.concat(measurements)
    all_df = all_df.astype({"ID": "ushort"})
    print(all_df)

    extracted_features = extract_features(
        all_df,
        column_id="ID",
        column_sort=data.TIME,
        show_warnings=True,
        impute_function=impute,
    )
    print("extracted_features.shape: ", extracted_features.shape)
    features_filtered = select_features(
        extracted_features,
        y,
        ml_task="classification",
        multiclass=True,
        show_warnings=True,
    )
    print("features_filtered.shape: ", features_filtered.shape)
