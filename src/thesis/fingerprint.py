from enum import Enum
import math
from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

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

PD_DIFF_MEAN = f"{PD_DIFF} Mean"
PD_DIFF_SKEW = f"{PD_DIFF} Skewness"
PD_DIFF_KURT = f"{PD_DIFF} Kurtosis"
PD_DIFF_WEIB_A = f"{PD_DIFF} Weibull A"

TD_MEDIAN = f"{TD} Median"

CORR_PD_DIFF_TO_PD = f"{CORR} {PD_DIFF} - PD"
CORR_NEXT_PD_TO_PD = f"{CORR} Next PD - PD"


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
    finger[PD_DIFF_KURT] = df[data.PD_DIFF].kurt()
    # FIXME workaround
    if math.isnan(finger[PD_DIFF_KURT]):
        finger[PD_DIFF_KURT] = 0.0
    finger[PD_DIFF_WEIB_A], _ = calc_weibull_params(df[data.PD_DIFF])

    finger[TD_MEDIAN] = df[data.TIME_DIFF].median()

    next_pd = df[data.PD][1:].reset_index(drop=True)
    # FIXME workaround
    if df[data.TIME_DIFF].sum() <= 60000:
        finger[CORR_PD_DIFF_TO_PD], _ = stats.pearsonr(next_pd, df[data.PD_DIFF][:-1])
        finger[CORR_NEXT_PD_TO_PD], _ = stats.pearsonr(df[data.PD][:-1], next_pd)
    else:
        finger[CORR_PD_DIFF_TO_PD] = _correlate_with_bins(
            next_pd, df[data.PD_DIFF][:-1]
        )
        finger[CORR_NEXT_PD_TO_PD] = _correlate_with_bins(df[data.PD][:-1], next_pd)
    assert not math.isnan(finger[CORR_PD_DIFF_TO_PD])
    assert not math.isnan(finger[CORR_NEXT_PD_TO_PD])

    return finger


def lukas_plus_tu_graz(df: pd.DataFrame) -> pd.Series:
    lukas_finger = lukas(df)
    return lukas_finger.combine_first(tu_graz(df))


def build_set(
    measurements: List[pd.DataFrame],
    fingerprint: Callable = tu_graz,
    add_class: bool = True,
) -> pd.DataFrame:
    fingers = pd.DataFrame([fingerprint(measurement) for measurement in measurements])
    if add_class:
        fingers[data.CLASS] = pd.Series(
            data.get_defects(measurements), dtype="category"
        )
    return fingers
