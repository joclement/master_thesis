from enum import Enum
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from tsfresh.feature_extraction.feature_calculators import (
    change_quantiles,
    count_above_mean,
    count_below_mean,
    longest_strike_below_mean,
    number_peaks,
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    ratio_value_number_to_time_series_length,
)

from . import data
from .util import get_memory

memory = get_memory()

PD = "PD-Value"
PD_DIFF = "PD-Diff"
TD = "TimeDiff"
CORR = "Correlate"


class Group(Enum):
    pd = PD
    pd_diff = PD_DIFF
    td = TD
    corr = CORR

    def __str__(self):
        return "%s" % self.value


# @note: parameter in TU Graz fingerprint
PD_VAR = f"{PD} Variance"
PD_SKEW = f"{PD} Skewness"
PD_KURT = f"{PD} Kurtosis"
PD_WEIB_A = f"{PD} Weibull A"
PD_WEIB_B = f"{PD} Weibull B"

PD_DIFF_WEIB_B = f"{PD_DIFF} Weibull B"

TD_MAX = f"{TD} Max"
TD_MEAN = f"{TD} Mean"
TD_MIN = f"{TD} Min"
TD_VAR = f"{TD} Variance"
TD_SKEW = f"{TD} Skewness"
TD_KURT = f"{TD} Kurtosis"
TD_WEIB_A = f"{TD} Weibull A"
TD_WEIB_B = f"{TD} Weibull B"


# @note: parameter in Lukas fingerprint
PDS_PER_SEC = "Number of PDs/sec"

PD_MEAN = f"{PD} Mean"
PD_MAX = f"{PD} Max"
PD_CV = f"{PD} std/mean"
PD_SUM = f"{PD} Sum"

PD_DIFF_MEAN = f"{PD_DIFF} Mean"
PD_DIFF_SKEW = f"{PD_DIFF} Skewness"
PD_DIFF_KURT = f"{PD_DIFF} Kurtosis"
PD_DIFF_WEIB_A = f"{PD_DIFF} Weibull A"

TD_MEDIAN = f"{TD} Median"

CORR_PD_DIFF_TO_PD = f"{CORR} {PD_DIFF} - PD"
CORR_NEXT_PD_TO_PD = f"{CORR} Next PD - PD"

# @note: parameter in own fingerprint
POLARITY = "+DC/-DC"


def keep_needed_columns(measurements: List[pd.DataFrame]):
    return [df[[data.TIME_DIFF, data.PD_DIFF, data.PD]] for df in measurements]


def get_parameter_group(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    wanted_columns = [column for column in df.columns if group.value in column]
    return df[wanted_columns].copy()


# TODO Issue #22: ensure that weibull fit is correct
@memory.cache
def calc_weibull_params(data: Union[list, pd.Series]) -> Tuple[float, float]:
    weibull_a, _, weibull_b = stats.weibull_min.fit(data, floc=0.0)
    return weibull_a, weibull_b


# Based on https://doi.org/10.5324/nordis.v0i26.3268
def tugraz(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_VAR] = df[data.PD].var()
    finger[PD_SKEW] = df[data.PD].skew()
    finger[PD_KURT] = df[data.PD].kurt()
    finger[PD_DIFF_WEIB_A], finger[PD_DIFF_WEIB_B] = calc_weibull_params(
        df[data.PD_DIFF]
    )

    finger[TD_MAX] = df[data.TIME_DIFF].max()
    finger[TD_MEAN] = df[data.TIME_DIFF].mean()
    finger[TD_MIN] = df[data.TIME_DIFF].min()
    finger[TD_VAR] = df[data.TIME_DIFF].var()
    finger[TD_SKEW] = df[data.TIME_DIFF].skew()
    finger[TD_KURT] = df[data.TIME_DIFF].kurt()
    finger[TD_WEIB_A], finger[TD_WEIB_B] = calc_weibull_params(df[data.TIME_DIFF][1:])

    finger[PDS_PER_SEC] = len(df[data.TIME_DIFF]) / (df[data.TIME_DIFF].sum() / 1000)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger


def _correlate_with_bins(x: pd.Series, y: pd.Series, num_of_boxes: int = 100):
    indices = pd.cut(x, num_of_boxes, labels=list(range(num_of_boxes)), precision=20)

    x_boxes = [[] for _ in range(num_of_boxes)]  # type: List[List[float]]
    y_boxes = [[] for _ in range(num_of_boxes)]  # type: List[List[float]]
    for idx, box_idx in enumerate(indices):
        x_boxes[box_idx].append(x[idx])
        y_boxes[box_idx].append(y[idx])

    x_means = np.array([np.mean(x_box) if len(x_box) > 0 else 0.0 for x_box in x_boxes])
    y_means = np.array([np.mean(y_box) if len(y_box) > 0 else 0.0 for y_box in y_boxes])

    correlation_coefficiient, _ = stats.pearsonr(x_means, y_means)
    assert -1 <= correlation_coefficiient <= 1
    return correlation_coefficiient


def ott(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_MEAN] = df[data.PD].mean()
    finger[PD_CV] = df[data.PD].std() / df[data.PD].mean()
    finger[PD_MAX] = df[data.PD].max()
    finger[PD_WEIB_A], finger[PD_WEIB_B] = calc_weibull_params(df[data.PD])

    finger[PD_DIFF_MEAN] = df[data.PD_DIFF].mean()
    finger[PD_DIFF_SKEW] = df[data.PD_DIFF].skew()
    finger[PD_DIFF_KURT] = df[data.PD_DIFF].kurt()
    finger[PD_DIFF_WEIB_A], _ = calc_weibull_params(df[data.PD_DIFF])

    finger[TD_MEDIAN] = df[data.TIME_DIFF].median()

    next_pd = df[data.PD][1:].reset_index(drop=True)
    finger[CORR_PD_DIFF_TO_PD] = _correlate_with_bins(df[data.PD], df[data.PD_DIFF])
    finger[CORR_NEXT_PD_TO_PD] = _correlate_with_bins(df[data.PD][:-1], next_pd)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger


def build_set(
    measurements: List[pd.DataFrame],
    fingerprint: Callable,
    add_class: bool = False,
) -> pd.DataFrame:
    fingers = pd.DataFrame([fingerprint(measurement) for measurement in measurements])
    if add_class:
        fingers[data.CLASS] = pd.Series(
            data.get_defects(measurements), dtype="category"
        )
    return fingers


def pd_mean(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].mean() for df in measurements]).reshape(-1, 1)


def pd_std(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].std() for df in measurements]).reshape(-1, 1)


def pd_median(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].median() for df in measurements]).reshape(-1, 1)


def pd_max(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].max() for df in measurements]).reshape(-1, 1)


def pd_min(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].min() for df in measurements]).reshape(-1, 1)


def pd_sum(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].sum() for df in measurements]).reshape(-1, 1)


def pd_var(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD].var() for df in measurements]).reshape(-1, 1)


def data_len(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([len(df.index) for df in measurements]).reshape(-1, 1)


def pd_number_peaks_50(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([number_peaks(df[data.PD], 50) for df in measurements]).reshape(
        -1, 1
    )


def pd_number_peaks_10(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([number_peaks(df[data.PD], 10) for df in measurements]).reshape(
        -1, 1
    )


def pd_ratio_value_number_to_time_series_length(
    measurements: List[pd.DataFrame],
) -> np.array:
    return np.array(
        [ratio_value_number_to_time_series_length(df[data.PD]) for df in measurements]
    ).reshape(-1, 1)


def pd_percentage_of_reoccurring_datapoints_to_all_datapoints(
    measurements: List[pd.DataFrame],
) -> np.array:
    return np.array(
        [
            percentage_of_reoccurring_datapoints_to_all_datapoints(df[data.PD])
            for df in measurements
        ]
    ).reshape(-1, 1)


def pd_count_below_mean(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([count_below_mean(df[data.PD]) for df in measurements]).reshape(
        -1, 1
    )


def pd_count_above_mean(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([count_above_mean(df[data.PD]) for df in measurements]).reshape(
        -1, 1
    )


def pd_change_quantiles(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [change_quantiles(df[data.PD], 0.0, 0.7, True, "mean") for df in measurements]
    ).reshape(-1, 1)


def pd_normed_weibull_a(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [
            calc_weibull_params(df[data.PD].sort_values() / df[data.PD].max())[0]
            for df in measurements
        ]
    ).reshape(-1, 1)


def pd_normed_weibull_b(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [
            calc_weibull_params(df[data.PD].sort_values() / df[data.PD].max())[1]
            for df in measurements
        ]
    ).reshape(-1, 1)


def td_kurt(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.TIME_DIFF].kurt() for df in measurements]).reshape(-1, 1)


def td_skew(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.TIME_DIFF].skew() for df in measurements]).reshape(-1, 1)


def td_median(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.TIME_DIFF].median() for df in measurements]).reshape(-1, 1)


def td_longest_strike_below_mean(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [longest_strike_below_mean(df[data.TIME_DIFF]) for df in measurements]
    ).reshape(-1, 1)


def td_change_quantiles(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [
            change_quantiles(df[data.TIME_DIFF], 0.0, 0.3, True, "var")
            for df in measurements
        ]
    ).reshape(-1, 1)


def td_weibull_a(measurements: List[pd.DataFrame]) -> np.array:
    return np.array(
        [
            calc_weibull_params(df[data.TIME_DIFF].cumsum() / df[data.TIME_DIFF].sum())[
                0
            ]
            for df in measurements
        ]
    ).reshape(-1, 1)


def td_sum(measurements: List[pd.DataFrame]) -> np.array:
    return np.array([df[data.PD_DIFF].sum() for df in measurements]).reshape(-1, 1)


def feature(feature: Callable):
    return feature.__name__, FunctionTransformer(feature)


def own_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            feature(pd_mean),
            feature(pd_std),
            feature(pd_median),
            feature(pd_max),
            feature(pd_min),
            feature(pd_sum),
            feature(pd_var),
            feature(data_len),
            feature(pd_number_peaks_50),
            feature(pd_number_peaks_10),
            feature(pd_ratio_value_number_to_time_series_length),
            feature(pd_percentage_of_reoccurring_datapoints_to_all_datapoints),
            feature(pd_count_below_mean),
            feature(pd_count_above_mean),
            feature(pd_change_quantiles),
            feature(pd_normed_weibull_a),
            feature(pd_normed_weibull_b),
            feature(td_kurt),
            feature(td_skew),
            feature(td_median),
            feature(td_longest_strike_below_mean),
            feature(td_change_quantiles),
            feature(td_weibull_a),
            feature(td_sum),
        ]
    )


def own(df: pd.DataFrame) -> pd.Series:
    own = own_feature_union().transform([df])
    finger = pd.Series(data=own, dtype=float)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger


def seqown(df: pd.DataFrame) -> pd.Series:
    own = [
        df[data.PD].mean(),
        df[data.PD].std(),
        df[data.PD].median(),
        df[data.PD].max(),
        df[data.PD].sum(),
        df[data.PD].var(),
        len(df.index),
        df[data.TIME_DIFF].skew(),
        df[data.TIME_DIFF].median(),
        number_peaks(df[data.TIME_DIFF], 3),
        df[data.TIME_DIFF].sum(),
    ]

    finger = pd.Series(data=own, dtype=float)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger
