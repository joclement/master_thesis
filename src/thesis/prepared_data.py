from typing import List, Tuple, Union

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import tsfresh.feature_extraction
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.transformers import RelevantFeatureAugmenter
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")


def convert_to_tsfresh_dataset(measurements: List[pd.DataFrame]) -> pd.DataFrame:
    measurements = [m.loc[:, [data.TIME, data.PD]] for m in measurements]
    for index, df in enumerate(measurements):
        df["id"] = index
        df["kind"] = data.PD
    all_df = pd.concat(measurements)
    all_df = all_df.rename(columns={data.PD: "value"})
    return all_df


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    df[data.TIME] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
    df.set_index(data.TIME, inplace=True)
    time_series = df[data.PD]
    return time_series.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()


def oned(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
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
    return X, None


def twod(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    for df in measurements:
        df.drop(df.columns.difference([data.TIME_DIFF, data.PD]), axis=1, inplace=True)
    min_len = min([len(m) for m in measurements])
    X = to_time_series_dataset(
        [df[: config["multiple_of_min_len"] * min_len] for df in measurements]
    )
    return X, None


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


def seqfinger_ott(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.lukas, duration)
            for df in measurements
        ]
    )
    return X, None


def seqfinger_tugraz(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.tu_graz, duration)
            for df in measurements
        ]
    )
    return X, None


def seqfinger_both(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    duration = pd.Timedelta(config["duration"])
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.lukas_plus_tu_graz, duration)
            for df in measurements
        ]
    )
    return X, None


def seqfinger_tsfresh(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    # TODO do own implementation
    return None, None


def finger_ott(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.lukas), MinMaxScaler()


def finger_own(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.own), MinMaxScaler()


def finger_tugraz(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    return fingerprint.build_set(measurements, fingerprint.tu_graz), MinMaxScaler()


def finger_both(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    fingerprint.keep_needed_columns(measurements)
    return (
        fingerprint.build_set(measurements, fingerprint.lukas_plus_tu_graz),
        MinMaxScaler(),
    )


def finger_tsfresh(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, Union[None, TransformerMixin]]:
    X_data = convert_to_tsfresh_dataset(measurements)
    if "default_fc_parameters" in config:
        DefaultFcParameters = getattr(
            tsfresh.feature_extraction, config["default_fc_parameters"]
        )
    else:
        DefaultFcParameters = ComprehensiveFCParameters
    tsfreshTransformer = RelevantFeatureAugmenter(
        column_id="id",
        column_kind="kind",
        column_sort=data.TIME,
        column_value="value",
        fdr_level=config["fdr_level"],
        ml_task="classification",
        multiclass=True,
        n_jobs=config["n_jobs"],
        default_fc_parameters=DefaultFcParameters(),
    )
    tsfreshTransformer.set_timeseries_container(X_data)
    X = pd.DataFrame(index=list(range(len(measurements))))
    return X, tsfreshTransformer
