from enum import Enum
import math
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
from pyts.multivariate.transformation import WEASELMUSE
from pyts.transformation import BOSS, WEASEL
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tslearn.utils import to_time_series_dataset

from . import data, fingerprint
from .constants import PART
from .data import PATH, PD, START_TIME, TIME_DIFF, VOLTAGE_SIGN
from .util import get_memory, to_dataTIME

MAX_FREQUENCY = pd.tseries.frequencies.to_offset("50us")

ONEPD_DURATION = pd.Timedelta("10 seconds")

memory = get_memory()


class NormalizationMethod(Enum):
    none = "none"
    zscore = "zscore"
    minmax = "minmax"


class MeasurementNormalizer(TransformerMixin, BaseEstimator):
    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        for df in measurements:
            df.loc[:, PD] /= df[PD].max()
        return measurements

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}


class TsfreshTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, tsfresh_data_path, **kw_args):
        self.set_tsfresh_data(tsfresh_data_path)

    def _get_feature_name(self, _):
        self._i += 1
        return self._i

    def set_tsfresh_data(self, tsfresh_data_path):
        self._i = 0
        self.tsfresh_data_path = tsfresh_data_path
        self._tsfresh_data = pd.read_csv(
            self.tsfresh_data_path, header=0, index_col=[PATH, PART]
        )
        self._tsfresh_data = self._tsfresh_data.rename(columns=self._get_feature_name)
        del self._i

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
            self.set_tsfresh_data(parameters["tsfresh_data_path"])
        return self


def tsfresh(**config):
    return TsfreshTransformer(config["tsfresh_data"])


def keep_needed_columns(measurements: List[pd.DataFrame]):
    return [df[[data.TIME_DIFF, data.PD]] for df in measurements]


class oned(BaseEstimator, TransformerMixin):
    def __init__(self, fix_duration: str, frequency: str, **kw_args):
        self.set_params(**{"fix_duration": fix_duration, "frequency": frequency})

    def to_time_series(self, df: pd.DataFrame) -> pd.Series:
        duration = df[data.TIME_DIFF].sum()
        assert duration <= self._fix_duration
        if duration < self._fix_duration:
            df = df.append(
                pd.DataFrame(
                    data={
                        data.PD: [0.0],
                        data.TIME_DIFF: [self._fix_duration - duration],
                    },
                    index=[len(df.index)],
                )
            )
        df.loc[:, "DateTimeIndex"] = pd.to_datetime(
            df[data.TIME_DIFF].cumsum(), unit=data.TIME_UNIT
        )
        df = df.set_index("DateTimeIndex")
        time_series = (
            df[data.PD]
            .asfreq(MAX_FREQUENCY, fill_value=0.0)
            .resample(self._frequency)
            .max()
        )
        if len(time_series.index) < self._time_series_len:
            len_diff = self._time_series_len - len(time_series.index)
            time_series = time_series.append(
                pd.Series(
                    index=[len(time_series.index) + i for i in range(len_diff)],
                    data=[0.0] * len_diff,
                )
            )
        elif len(time_series.index) > self._time_series_len:
            time_series = time_series.iloc[:-1]
        return time_series

    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return to_time_series_dataset(
            [self.to_time_series(df) for df in keep_needed_columns(measurements)]
        )

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}

    def get_params(self, deep=True):
        return {"fix_duration": self.fix_duration_str, "frequency": self.frequency_str}

    def set_params(self, **parameters):
        if "fix_duration" in parameters:
            self.fix_duration_str = parameters["fix_duration"]
            self._fix_duration = to_dataTIME(pd.Timedelta(self.fix_duration_str))
        if "frequency" in parameters:
            self.frequency_str = parameters["frequency"]
            self._frequency = pd.tseries.frequencies.to_offset(self.frequency_str)
        self._time_series_len = int(
            pd.Timedelta(self.fix_duration_str) / self._frequency
        )
        return self


def oned_func(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    return oned(config["fix_duration"], config["frequency"]).transform(measurements)


class Reshaper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return np.reshape(X, (X.shape[0], -1))


def oned_weasel(**config):
    return Pipeline(
        [
            ("oned", oned(**config["oned"])),
            (("reshaper", Reshaper())),
            ("weasel", WEASEL(**config["weasel"])),
        ]
    )


def oned_boss(**config):
    return Pipeline(
        [
            ("oned", oned(**config["oned"])),
            (("reshaper", Reshaper())),
            ("boss", BOSS(**config["boss"])),
        ]
    )


class twod(BaseEstimator, TransformerMixin):
    def __init__(self, max_len: int, normalize: str, **kw_args):
        self.set_params(**{"max_len": max_len, "normalize": normalize})

    def fit(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        return self

    def transform(self, measurements: List[pd.DataFrame], y=None, **kwargs):
        measurements = keep_needed_columns(measurements)
        if self._normalize is NormalizationMethod.minmax:
            for df in measurements:
                df.loc[:, PD] /= df[PD].max()
        elif self._normalize is NormalizationMethod.zscore:
            for df in measurements:
                df.loc[:, PD] = df.apply(zscore)[PD]
        return to_time_series_dataset([df[: self.max_len] for df in measurements])

    def get_params(self, deep=True):
        return {"normalize": self.normalize_str, "max_len": self.max_len}

    def set_params(self, **parameters):
        if "max_len" in parameters:
            self.max_len = parameters["max_len"]
        if "normalize" in parameters:
            self.normalize_str = parameters["normalize"]
            self._normalize = NormalizationMethod(self.normalize_str)
        return self


class AxisSwapper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return np.swapaxes(X, 1, 2)


def weaselmuse(**config):
    return Pipeline(
        [
            ("twod", twod(**config["twod"])),
            ("axisswapper", AxisSwapper()),
            ("weaselmuse", WEASELMUSE(**config["weaselmuse"])),
        ]
    )


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
            part.attrs[START_TIME] = index * int_dur + df.attrs[START_TIME]
            sequence.append(part.drop(columns="tmp_time"))

    return sequence


def split_by_durations(
    measurements: List[pd.DataFrame],
    max_duration: pd.Timedelta,
    step_duration: Optional[pd.Timedelta] = None,
    drop_empty=False,
) -> List[pd.DataFrame]:
    if step_duration is not None:
        if max_duration % step_duration != pd.Timedelta(0):
            raise ValueError(
                f"max_duration '{max_duration}' and "
                f"step_duration '{step_duration}' don't fit"
            )
        length = int(max_duration / step_duration)
        duration = step_duration
    else:
        duration = max_duration
    splitted_measurements = []
    for df in measurements:
        splitted_measurements.extend(
            _split_by_duration(df, duration, True, drop_empty=drop_empty)
        )
    if step_duration is None:
        return splitted_measurements
    stepped_measurements = []
    for idx in range(0, len(splitted_measurements) - length + 1):
        df = pd.concat(splitted_measurements[idx : idx + length])
        df = df.reset_index(drop=True)
        df.attrs = splitted_measurements[idx].attrs
        df.attrs[PART] = idx
        stepped_measurements.append(df)
    return stepped_measurements


def adapt_durations(
    measurements: List[pd.DataFrame],
    min_duration: str = "60 seconds",
    max_duration: str = "60 seconds",
    step_duration: Optional[str] = None,
    min_len: Optional[int] = None,
    split: bool = True,
    drop_empty: bool = True,
):
    min_dur = to_dataTIME(pd.Timedelta(min_duration))
    long_enough_measurements = [
        df for df in measurements if df[data.TIME_DIFF].sum() > min_dur
    ]
    if min_len is not None:
        long_enough_measurements = [
            df for df in long_enough_measurements if len(df.index) > min_len
        ]

    if len(long_enough_measurements) == 0:
        raise ValueError("No long enough data.")
    if not split:
        return long_enough_measurements
    return split_by_durations(
        long_enough_measurements,
        pd.Timedelta(max_duration),
        step_duration=pd.Timedelta(step_duration) if step_duration else None,
        drop_empty=drop_empty,
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
