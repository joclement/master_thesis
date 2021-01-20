import math
from typing import List
import warnings

import pandas as pd
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint
from .util import to_dataTIME

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")

PART = "part"


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    df["DateTimeIndex"] = pd.to_datetime(
        df[data.TIME_DIFF].cumsum(), unit=data.TIME_UNIT
    )
    df.set_index("DateTimeIndex", inplace=True)
    time_series = df[data.PD]
    return time_series.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()


def oned(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    time_serieses = [
        _convert_to_time_series(df, config["frequency"]) for df in measurements
    ]
    return to_time_series_dataset(time_serieses)


def twod(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    for df in measurements:
        df.drop(df.columns.difference([data.TIME_DIFF, data.PD]), axis=1, inplace=True)
    return to_time_series_dataset(measurements)


def _split_by_duration(
    df: pd.DataFrame, duration: pd.Timedelta, drop_last: bool, drop_empty: bool = False
) -> List[pd.DataFrame]:
    if drop_last:
        end_edge = math.ceil(df[data.TIME_DIFF].sum())
    else:
        ratio = df[data.TIME_DIFF].sum() / to_dataTIME(duration)
        end_edge = math.floor((ratio + 1) * to_dataTIME(duration))
    bins = range(0, end_edge, to_dataTIME(duration))
    groups = df.groupby(pd.cut(df[data.TIME_DIFF].cumsum(), bins))
    sequence = []
    for index, group in enumerate(groups):
        part = group[1]
        if len(part.index) == 0 and not drop_empty:
            warnings.warn(f"Empty Part in data for duration {duration}.")
        if not drop_empty or len(part.index) > 0:
            part.attrs[PART] = index
            sequence.append(part.reset_index(drop=True))

    return sequence


def split_by_durations(
    measurements: List[pd.DataFrame], max_duration: pd.Timedelta, drop_empty=False
) -> List[pd.DataFrame]:
    splitted_measurements = []
    for df in measurements:
        splitted_measurements.extend(
            _split_by_duration(df, max_duration, True, drop_empty=drop_empty)
        )
    return splitted_measurements


def _build_fingerprint_sequence(
    df: pd.DataFrame, finger_algo, duration: pd.Timedelta, step_duration: pd.Timedelta
):
    if duration % step_duration != pd.Timedelta(0):
        raise ValueError(
            f"duration '{duration}' and step_duration '{step_duration}' don't fit"
        )
    length = int(duration / step_duration)
    step_sequence = _split_by_duration(df, step_duration, False)

    sequence = []
    for idx in range(0, len(step_sequence) - length + 1):
        sub_df = pd.concat(step_sequence[idx : idx + length])
        sequence.append(sub_df)

    # FIXME workaround
    if len(sequence[-1].index) < 3:
        del sequence[-1]
    assert all([len(sub_df.index) >= 3 for sub_df in sequence])
    assert all([not sub_df.index.isnull().any() for sub_df in sequence])
    assert len(sequence) >= 3

    fingerprint.keep_needed_columns(sequence)
    return fingerprint.build_set(sequence, finger_algo).to_numpy()


def seqfinger_ott(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    duration = pd.Timedelta(config["duration"])
    step_duration = pd.Timedelta(config["step_duration"])

    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.lukas, duration, step_duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_own(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    duration = pd.Timedelta(config["duration"])
    step_duration = pd.Timedelta(config["step_duration"])

    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.own, duration, step_duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_tugraz(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    duration = pd.Timedelta(config["duration"])
    step_duration = pd.Timedelta(config["step_duration"])

    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(
                df, fingerprint.tu_graz, duration, step_duration
            )
            for df in measurements
        ]
    )
    return X


def seqfinger_both(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    duration = pd.Timedelta(config["duration"])
    step_duration = pd.Timedelta(config["step_duration"])

    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(
                df, fingerprint.lukas_plus_tu_graz, duration, step_duration
            )
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
