from typing import List

import pandas as pd
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    df["DateTimeIndex"] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
    df.set_index("DateTimeIndex", inplace=True)
    time_series = df[data.PD]
    return time_series.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()


def oned(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    time_serieses = [
        _convert_to_time_series(df, config["frequency"]) for df in measurements
    ]
    min_len = min([len(time_series) for time_series in time_serieses])
    X = to_time_series_dataset(
        [
            time_series[: config["multiple_of_min_len"] * min_len]
            for time_series in time_serieses
        ]
    )
    return X


def _shorten(measurements: List[pd.DataFrame], multiple_of_min_len: int):
    max_len = multiple_of_min_len * min([len(m) for m in measurements])
    return [df.loc[:max_len, :] for df in measurements]


def twod(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    for df in measurements:
        df.drop(df.columns.difference([data.TIME_DIFF, data.PD]), axis=1, inplace=True)
    return to_time_series_dataset(_shorten(measurements, config["multiple_of_min_len"]))


def _build_fingerprint_sequence(df: pd.DataFrame, finger_algo, duration: pd.Timedelta):
    timedelta_sum = pd.Timedelta(0)
    index_sequence_splits = []
    for index, value in df[data.TIME_DIFF].iteritems():
        timedelta_sum += pd.Timedelta(value, unit=data.TIME_UNIT)
        if timedelta_sum >= duration:
            index_sequence_splits.append(index)
            timedelta_sum = pd.Timedelta(0)
    sequence = [df.iloc[: index_sequence_splits[0]]]
    for idx in range(1, len(index_sequence_splits)):
        sequence.append(
            df.iloc[index_sequence_splits[idx - 1] : index_sequence_splits[idx]]
        )

    too_short_indexes = []
    for index, sub_df in enumerate(sequence):
        if len(sub_df.index) <= 2:
            too_short_indexes.append(index)

    if len(too_short_indexes) > 0:
        if (
            len(too_short_indexes) - len(range(too_short_indexes[0], len(sequence)))
            <= 1
        ):
            del sequence[too_short_indexes[0] :]
        else:
            raise ValueError("Invalid measurement file.")

    assert all([len(sub_df.index) >= 3 for sub_df in sequence])
    assert all([not sub_df.index.isnull().any() for sub_df in sequence])
    # FIXME adapt so that minimum will be 4
    assert len(sequence) >= 3

    return fingerprint.build_set(sequence, finger_algo).to_numpy()


def seqfinger_ott(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    _shorten(measurements, config["multiple_of_min_len"])
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.lukas, duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_own(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    _shorten(measurements, config["multiple_of_min_len"])
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.own, duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_tugraz(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    _shorten(measurements, config["multiple_of_min_len"])
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.tu_graz, duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_both(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    _shorten(measurements, config["multiple_of_min_len"])
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.lukas_plus_tu_graz, duration)
            for df in measurements
        ]
    )
    return X


def finger_ott(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.lukas)


def finger_own(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.own)


def finger_tugraz(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.tu_graz)


def finger_both(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.lukas_plus_tu_graz)
