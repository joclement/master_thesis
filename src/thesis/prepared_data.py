import math
from typing import List
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint
from .constants import PART
from .data import PATH, START_TIME, TIME_DIFF, VOLTAGE_SIGN
from .util import get_memory, to_dataTIME

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")

ONEPD_DURATION = pd.Timedelta("10 seconds")

memory = get_memory()


class TsfreshTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, **config):
        self.tsfresh_data = pd.read_csv(
            config["tsfresh_data"], header=0, index_col=[PATH, PART]
        )

    def fit(self, X: List[pd.DataFrame], y=None, **kwargs):
        self.n_features_in_ = len(self.tsfresh_data.columns)
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        wanted_rows = [fingerprint.get_X_index(df) for df in measurements]
        tsfresh_data = self.tsfresh_data.loc[wanted_rows, :]
        tsfresh_data[fingerprint.POLARITY] = [
            df.attrs[VOLTAGE_SIGN] for df in measurements
        ]
        return tsfresh_data

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}


def tsfresh(**config):
    return TsfreshTransformer(**config)


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    df.loc[:, "DateTimeIndex"] = pd.to_datetime(
        df[data.TIME_DIFF].cumsum(), unit=data.TIME_UNIT
    )
    df = df.set_index("DateTimeIndex")
    time_series = df[data.PD]
    return time_series.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()


def keep_needed_columns(measurements: List[pd.DataFrame]):
    return [df[[data.TIME_DIFF, data.PD]] for df in measurements]


def oned_func(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    fix_duration = to_dataTIME(pd.Timedelta(config["fix_duration"]))
    measurements = keep_needed_columns(measurements)

    # FIXME workaround due to _split_by_durations bug
    equal_lenghted_dfs = []
    for df in measurements:
        duration = df[data.TIME_DIFF].sum()
        if duration > fix_duration:
            df = df[df[data.TIME_DIFF].cumsum() <= fix_duration]
        elif duration < fix_duration:
            df = df.append(
                pd.DataFrame(
                    data={data.PD: [0.0], data.TIME_DIFF: [fix_duration - duration]},
                    index=[len(df.index)],
                )
            )
        equal_lenghted_dfs.append(df)
    measurements = equal_lenghted_dfs

    time_serieses = [
        _convert_to_time_series(df, config["frequency"]) for df in measurements
    ]
    return to_time_series_dataset(time_serieses)


def oned(**config) -> FunctionTransformer:
    return FunctionTransformer(oned_func, kw_args=config)


def twod_func(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    measurements = keep_needed_columns(measurements)
    return to_time_series_dataset([m[: config["max_len"]] for m in measurements])


def twod(**config) -> FunctionTransformer:
    return FunctionTransformer(twod_func, kw_args=config)


# FIXME _split_by_duration does not work properly, e.g. for oned
def _split_by_duration(
    df: pd.DataFrame, duration: pd.Timedelta, drop_last: bool, drop_empty: bool = False
) -> List[pd.DataFrame]:
    df = df.assign(**{"tmp_time": df[TIME_DIFF].cumsum()})
    int_dur = to_dataTIME(duration)
    if drop_last:
        end_edge = math.ceil(df["tmp_time"].iloc[-1])
    else:
        ratio = df["tmp_time"].iloc[-1] / int_dur
        end_edge = math.floor((ratio + 1) * int_dur)
    bins = range(0, end_edge, int_dur)
    groups = df.groupby(pd.cut(df["tmp_time"], bins))
    sequence = []
    for index, group in enumerate(groups):
        part = group[1]
        if len(part.index) == 0 and not drop_empty:
            warnings.warn(f"Empty Part in data for duration {duration}.")
        if not drop_empty or len(part.index) >= duration / ONEPD_DURATION:
            part = part.reset_index(drop=True)
            if (
                drop_empty
                and len(part) > 0
                and part[TIME_DIFF][0] > part["tmp_time"][0] % int_dur
            ):
                correct_timediff = part["tmp_time"][0] % int_dur
                part.loc[0, TIME_DIFF] = correct_timediff
            part.attrs[PART] = index
            part.attrs[PATH] = df.attrs[PATH]
            part.attrs[START_TIME] = index * int_dur + df.attrs[START_TIME]
            sequence.append(part.drop(columns="tmp_time"))

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


def adapt_durations(
    measurements: List[pd.DataFrame],
    min_duration: str = "60 seconds",
    max_duration: str = "60 seconds",
    split: bool = True,
    drop_empty: bool = True,
):
    min_duration = pd.Timedelta(min_duration)
    long_enough_measurements = []
    for df in measurements:
        if df[data.TIME_DIFF].sum() > to_dataTIME(min_duration):
            long_enough_measurements.append(df)

    if not split:
        return long_enough_measurements
    return split_by_durations(
        long_enough_measurements, pd.Timedelta(max_duration), drop_empty
    )


@memory.cache
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

    assert all([len(sub_df.index) >= 3 for sub_df in sequence])
    assert all([not sub_df.index.isnull().any() for sub_df in sequence])
    assert len(sequence) >= 3

    return pd.DataFrame([finger_algo(part) for part in sequence])


def seqfinger_seqown_func(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    duration = pd.Timedelta(config["duration"])
    step_duration = pd.Timedelta(config["step_duration"])

    measurements = fingerprint.keep_needed_columns(measurements)
    X = to_time_series_dataset(
        [
            _build_fingerprint_sequence(df, fingerprint.seqown, duration, step_duration)
            for df in measurements
        ]
    )
    return X


def seqfinger_seqown(**config) -> FunctionTransformer:
    return FunctionTransformer(seqfinger_seqown_func, kw_args=config)


def finger_ott(**config) -> TransformerMixin:
    return fingerprint.ott_feature_union(**config)


def finger_own(**config) -> TransformerMixin:
    return fingerprint.own_feature_union(**config)


def finger_tugraz(**config) -> TransformerMixin:
    return fingerprint.tugraz_feature_union(**config)


def do_nothing(extracted_features: pd.DataFrame, **config):
    return extracted_features


def finger_all(**config) -> TransformerMixin:
    return FunctionTransformer(do_nothing, kw_args=config)


def extract_features(
    measurements: List[pd.DataFrame],
) -> pd.DataFrame:
    return pd.concat([fingerprint.extract_features(df) for df in measurements])
