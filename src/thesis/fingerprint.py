from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from . import data

PD_VAR = "PD Variance"
PD_SKEW = "PD Skewness"
PD_KURT = "PD Kurtosis"
PD_WEIB_A = "PD Weibull A"
PD_WEIB_B = "PD Weibull B"
PD_DIFF_WEIB_B = "PD Diff Weibull B"
TD_MAX = "TimeDiff Max"
TD_MEAN = "TimeDiff Mean"
TD_MIN = "TimeDiff Min"
TD_VAR = "TimeDiff Variance"
TD_SKEW = "TimeDiff Skewness"
TD_KURT = "TimeDiff Kurtosis"
TD_WEIB_A = "TimeDiff Weibull A"
TD_WEIB_B = "TimeDiff Weibull B"
PDS_PER_SEC = "Number of PDs/sec"

PD_MEAN = "PD Mean"
PD_MAX = "PD Max"
PD_DIFF_MEAN = "PD Diff Mean"
PD_DIFF_SKEW = "PD Diff Skewness"
PD_DIFF_KURT = "PD Diff Kurtosis"
PD_DIFF_WEIB_A = "PD Diff Weibull A"
TD_MEDIAN = "TimeDiff Median"
CORR_PD_DIFF_TO_PD = "Correlate PD Diff - PD"
CORR_NEXT_PD_TO_PD = "Correlate Next PD - PD"


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

    finger[PD_DIFF_WEIB_A], finger[PD_DIFF_WEIB_B] = calc_weibull_params(_pd_diff(df))

    finger[TD_MAX] = df[data.TIMEDIFF].max()
    finger[TD_MEAN] = df[data.TIMEDIFF].mean()
    finger[TD_MIN] = df[data.TIMEDIFF].min()
    finger[TD_VAR] = df[data.TIMEDIFF].var()
    finger[TD_SKEW] = df[data.TIMEDIFF].skew()
    finger[TD_KURT] = df[data.TIMEDIFF].kurt()
    finger[TD_WEIB_A], finger[TD_WEIB_B] = calc_weibull_params(df[data.TIMEDIFF][1:])

    finger[PDS_PER_SEC] = len(df[data.TIMEDIFF]) / df[data.TIMEDIFF].sum()

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


def _pd_diff(df: pd.DataFrame):
    return df[data.PD].diff()[1:].reset_index(drop=True).abs()


def lukas(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_MEAN] = df[data.PD].mean()
    finger[PD_VAR] = df[data.PD].var()
    finger[PD_MAX] = df[data.PD].max()
    finger[PD_WEIB_A], finger[PD_WEIB_B] = calc_weibull_params(df[data.PD])

    pd_diff = _pd_diff(df)
    finger[PD_DIFF_MEAN] = pd_diff.mean()
    finger[PD_DIFF_SKEW] = pd_diff.skew()
    finger[PD_DIFF_KURT] = pd_diff.kurt()
    finger[PD_DIFF_WEIB_A], _ = calc_weibull_params(pd_diff)

    finger[TD_MEDIAN] = df[data.TIMEDIFF].median()

    next_pd = df[data.PD][1:].reset_index(drop=True)
    finger[CORR_PD_DIFF_TO_PD] = _correlate_with_bins(next_pd, pd_diff)
    finger[CORR_NEXT_PD_TO_PD] = _correlate_with_bins(df[data.PD][:-1], next_pd)

    return finger


def lukas_plus_tu_graz(df: pd.DataFrame) -> pd.Series:
    lukas_finger = lukas(df)
    return lukas_finger.combine_first(tu_graz(df))


def build_set(measurements: list, fingerprint: Callable = tu_graz) -> pd.DataFrame:
    fingers = pd.DataFrame([fingerprint(measurement) for measurement in measurements])
    fingers[data.CLASS] = pd.Series(data.get_defects(measurements), dtype="category")
    return fingers
