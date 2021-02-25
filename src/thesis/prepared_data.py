import math
from typing import List
import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint
from .constants import PART
from .data import PATH, PD, START_TIME, TIME_DIFF, VOLTAGE_SIGN
from .util import get_memory, to_dataTIME

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")

ONEPD_DURATION = pd.Timedelta("10 seconds")

memory = get_memory()


class MeasurementNormalizer(TransformerMixin, BaseEstimator):
    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        for df in measurements:
            df[PD] /= df[PD].max()
        return measurements

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}


class TsfreshTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, tsfresh_data_path, **kw_args):
        self.tsfresh_data_path = tsfresh_data_path
        self._tsfresh_data = pd.read_csv(
            tsfresh_data_path, header=0, index_col=[PATH, PART]
        )

    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        self.n_features_in_ = len(self._tsfresh_data.columns)
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        wanted_rows = [fingerprint.get_X_index(df) for df in measurements]
        tsfresh_data = self._tsfresh_data.loc[wanted_rows, :]
        tsfresh_data[fingerprint.POLARITY] = pd.Series(
            data=[df.attrs[VOLTAGE_SIGN] for df in measurements],
            index=wanted_rows,
        )
        return tsfresh_data

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}

    def get_params(self, deep=True):
        return {"tsfresh_data_path": str(self.tsfresh_data_path)}

    def set_params(self, **parameters):
        if "tsfresh_data_path" in parameters:
            self.tsfresh_data_path = parameters["tsfresh_data_path"]
            self._tsfresh_data = pd.read_csv(
                self.tsfresh_data_path, header=0, index_col=[PATH, PART]
            )
        return self


def tsfresh(**config):
    return TsfreshTransformer(config["tsfresh_data"])


def _convert_to_time_series(df: pd.DataFrame, frequency) -> pd.Series:
    df.loc[:, "DateTimeIndex"] = pd.to_datetime(
        df[data.TIME_DIFF].cumsum(), unit=data.TIME_UNIT
    )
    df = df.set_index("DateTimeIndex")
    time_series = df[data.PD]
    return time_series.asfreq(MAX_FREQUENCY, fill_value=0.0).resample(frequency).max()


def keep_needed_columns(measurements: List[pd.DataFrame]):
    return [df[[data.TIME_DIFF, data.PD]] for df in measurements]


class oned(BaseEstimator, TransformerMixin):
    def __init__(self, fix_duration: str, frequency: str, **kw_args):
        self.fix_duration_str = fix_duration
        self._fix_duration = to_dataTIME(pd.Timedelta(self.fix_duration_str))
        self.frequency = frequency

    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        measurements = keep_needed_columns(measurements)

        # FIXME workaround due to _split_by_durations bug
        equal_lenghted_dfs = []
        for df in measurements:
            duration = df[data.TIME_DIFF].sum()
            if duration > self._fix_duration:
                df = df[df[data.TIME_DIFF].cumsum() <= self._fix_duration]
            elif duration < self._fix_duration:
                df = df.append(
                    pd.DataFrame(
                        data={
                            data.PD: [0.0],
                            data.TIME_DIFF: [self._fix_duration - duration],
                        },
                        index=[len(df.index)],
                    )
                )
            equal_lenghted_dfs.append(df)
        measurements = equal_lenghted_dfs

        return to_time_series_dataset(
            [_convert_to_time_series(df, self.frequency) for df in measurements]
        )

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}

    def get_params(self, deep=True):
        return {"fix_duration": self.fix_duration_str, "frequency": self.frequency}

    def set_params(self, **parameters):
        if "fix_duration" in parameters:
            self.fix_duration_str = parameters["fix_duration"]
            self._fix_duration = to_dataTIME(pd.Timedelta(self.fix_duration_str))
        if "frequency" in parameters:
            self.frequency = parameters["frequency"]
        return self


def oned_func(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    return oned(config["fix_duration"], config["frequency"]).transform(measurements)


class twod(BaseEstimator, TransformerMixin):
    def __init__(self, max_len: int, **kw_args):
        self.max_len = max_len

    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        measurements = keep_needed_columns(measurements)
        return to_time_series_dataset([df[: self.max_len] for df in measurements])


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
