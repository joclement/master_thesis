from enum import Enum
import math
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
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
    for df in measurements:
        df.drop(
            df.columns.difference([data.TIME_DIFF, data.PD_DIFF, data.PD]),
            axis=1,
            inplace=True,
        )


def get_parameter_group(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    wanted_columns = [column for column in df.columns if group.value in column]
    return df[wanted_columns].copy()


# TODO Issue #22: ensure that weibull fit is correct
def calc_weibull_params(data: Union[list, pd.Series]) -> Tuple[float, float]:
    weibull_a, _, weibull_b = stats.weibull_min.fit(data, floc=0.0)
    return weibull_a, weibull_b


# Based on https://doi.org/10.5324/nordis.v0i26.3268
def tu_graz(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_VAR] = df[data.PD].var()
    finger[PD_SKEW] = df[data.PD].skew()
    finger[PD_KURT] = df[data.PD].kurt()
    # FIXME workaround
    if math.isnan(finger[PD_KURT]):
        finger[PD_KURT] = 0.0

    finger[PD_DIFF_WEIB_A], finger[PD_DIFF_WEIB_B] = calc_weibull_params(
        df[data.PD_DIFF]
    )

    finger[TD_MAX] = df[data.TIME_DIFF].max()
    finger[TD_MEAN] = df[data.TIME_DIFF].mean()
    finger[TD_MIN] = df[data.TIME_DIFF].min()
    finger[TD_VAR] = df[data.TIME_DIFF].var()
    finger[TD_SKEW] = df[data.TIME_DIFF].skew()
    finger[TD_KURT] = df[data.TIME_DIFF].kurt()
    # FIXME workaround
    if math.isnan(finger[TD_KURT]):
        finger[TD_KURT] = 0.0
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


def lukas(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_MEAN] = df[data.PD].mean()
    finger[PD_CV] = df[data.PD].std() / df[data.PD].mean()
    finger[PD_MAX] = df[data.PD].max()
    finger[PD_WEIB_A], finger[PD_WEIB_B] = calc_weibull_params(df[data.PD])

    finger[PD_DIFF_MEAN] = df[data.PD_DIFF].mean()
    finger[PD_DIFF_SKEW] = df[data.PD_DIFF].skew()
    # FIXME workaround
    if math.isnan(finger[PD_DIFF_SKEW]):
        finger[PD_DIFF_SKEW] = 0.0
    finger[PD_DIFF_KURT] = df[data.PD_DIFF].kurt()
    # FIXME workaround
    if math.isnan(finger[PD_DIFF_KURT]):
        finger[PD_DIFF_KURT] = 0.0
    finger[PD_DIFF_WEIB_A], _ = calc_weibull_params(df[data.PD_DIFF])

    finger[TD_MEDIAN] = df[data.TIME_DIFF].median()

    next_pd = df[data.PD][1:].reset_index(drop=True)
    # FIXME workaround
    assert data.TIME_UNIT == "ms"
    if df[data.TIME_DIFF].sum() <= 60000:
        finger[CORR_PD_DIFF_TO_PD], _ = stats.pearsonr(df[data.PD], df[data.PD_DIFF])
        # FIXME workaround
        if math.isnan(finger[CORR_PD_DIFF_TO_PD]):
            finger[CORR_PD_DIFF_TO_PD] = 0.0
        finger[CORR_NEXT_PD_TO_PD], _ = stats.pearsonr(df[data.PD][:-1], next_pd)
        # FIXME workaround
        if math.isnan(finger[CORR_NEXT_PD_TO_PD]):
            finger[CORR_NEXT_PD_TO_PD] = 0.0
    else:
        finger[CORR_PD_DIFF_TO_PD] = _correlate_with_bins(df[data.PD], df[data.PD_DIFF])
        finger[CORR_NEXT_PD_TO_PD] = _correlate_with_bins(df[data.PD][:-1], next_pd)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger


def lukas_plus_tu_graz(df: pd.DataFrame) -> pd.Series:
    lukas_finger = lukas(df)
    return lukas_finger.combine_first(tu_graz(df))


def build_set(
    measurements: List[pd.DataFrame],
    fingerprint: Callable = tu_graz,
    add_class: bool = False,
) -> pd.DataFrame:
    fingers = pd.DataFrame([fingerprint(measurement) for measurement in measurements])
    if add_class:
        fingers[data.CLASS] = pd.Series(
            data.get_defects(measurements), dtype="category"
        )
    return fingers


def own(df: pd.DataFrame) -> pd.Series:
    own = [
        df[data.PD].mean(),
        df[data.PD].std(),
        df[data.PD].median(),
        df[data.PD].max(),
        df[data.PD].min(),
        df[data.PD].sum(),
        df[data.PD].var(),
        len(df.index),
        number_peaks(df[data.PD], 50),
        number_peaks(df[data.PD], 10),
        ratio_value_number_to_time_series_length(df[data.PD]),
        percentage_of_reoccurring_datapoints_to_all_datapoints(df[data.PD]),
        count_below_mean(df[data.PD]),
        count_above_mean(df[data.PD]),
        change_quantiles(df[data.PD], 0.0, 0.7, True, "mean"),
        df[data.TIME_DIFF].kurt(),
        df[data.TIME_DIFF].skew(),
        df[data.TIME_DIFF].median(),
        longest_strike_below_mean(df[data.TIME_DIFF]),
        change_quantiles(df[data.TIME_DIFF], 0.0, 0.3, True, "var"),
        *calc_weibull_params(df[data.PD].sort_values() / df[data.PD].max()),
    ]

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
    ]

    finger = pd.Series(data=own, dtype=float)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger
