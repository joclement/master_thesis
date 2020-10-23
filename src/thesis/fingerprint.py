import math
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats

from . import data

PD_VAR = "PD Variance"
PD_SKEW = "PD Skewness"
PD_KURT = "PD Kurtosis"
PD_WEIB_A = "PD Weibull A"
PD_WEIB_B = "PD Weibull B"
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


def tu_graz(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_VAR] = df[data.PD].var()
    # TODO Issue #23: ensure that skewness is not 0 because of numerical problem
    finger[PD_SKEW] = df[data.PD].skew()
    # TODO Issue #23: ensure that skewness is not 0 because of numerical problem
    finger[PD_KURT] = df[data.PD].kurt()
    finger[PD_WEIB_A], finger[PD_WEIB_B] = calc_weibull_params(df[data.PD])

    finger[TD_MAX] = df[data.TIMEDIFF].max()
    finger[TD_MEAN] = df[data.TIMEDIFF].mean()
    finger[TD_MIN] = df[data.TIMEDIFF].min()
    finger[TD_VAR] = df[data.TIMEDIFF].var()
    finger[TD_SKEW] = df[data.TIMEDIFF].skew()
    finger[TD_KURT] = df[data.TIMEDIFF].kurt()
    finger[TD_WEIB_A], finger[TD_WEIB_B] = calc_weibull_params(df[data.TIMEDIFF][1:])

    finger[PDS_PER_SEC] = len(df[data.TIME]) / df[data.TIME].max()

    return finger


def _correlate_pd_and_pd_diff(pds, pd_diffs):
    indices = pd.cut(pds, 100, labels=list(range(100)), precision=20)

    pd_boxes = [[] for _ in range(100)]
    pd_diff_boxes = [[] for _ in range(100)]
    for idx, box_idx in enumerate(indices):
        pd_boxes[box_idx].append(pds[idx])
        pd_diff_boxes[box_idx].append(pd_diffs[idx])

    if any([len(pd_box) == 0 for pd_box in pd_boxes]):
        raise ValueError("Correlation can not not be computed: Too few data points")
    pd_means = [np.mean(pd_box) for pd_box in pd_boxes]
    pd_diff_means = [np.mean(pd_diff_box) for pd_diff_box in pd_diff_boxes]

    correlation_coefficiient, _ = stats.pearsonr(pd_means, pd_diff_means)
    if math.isnan(correlation_coefficiient):
        raise ValueError("Correlation between PD and Next PD could not be computed.")
    return correlation_coefficiient


def _correlate_pd_and_next_pd(pds, next_pds):
    indices = pd.cut(pds, 100, labels=list(range(100)), precision=20)

    pd_boxes = [[] for _ in range(100)]
    next_pd_boxes = [[] for _ in range(100)]
    for idx, box_idx in enumerate(indices):
        pd_boxes[box_idx].append(pds[idx])
        next_pd_boxes[box_idx].append(next_pds[idx])

    pd_means = [np.mean(pd_box) for pd_box in pd_boxes]
    next_pd_means = [np.mean(next_pd_box) for next_pd_box in next_pd_boxes]

    correlation_coefficiient, _ = stats.pearsonr(pd_means, next_pd_means)
    if math.isnan(correlation_coefficiient):
        raise ValueError("Correlation between PD and Next PD could not be computed.")
    return correlation_coefficiient


def lukas(df: pd.DataFrame) -> pd.Series:
    finger = pd.Series(dtype=float)

    finger[PD_MEAN] = df[data.PD].mean()
    finger[PD_VAR] = df[data.PD].var()
    finger[PD_MAX] = df[data.PD].max()
    finger[PD_WEIB_A], finger[PD_WEIB_B] = calc_weibull_params(df[data.PD])

    pd_diff = df[data.PD].diff()[1:].reset_index(drop=True)
    finger[PD_DIFF_MEAN] = pd_diff.mean()
    finger[PD_DIFF_SKEW] = pd_diff.skew()
    finger[PD_DIFF_KURT] = pd_diff.kurt()
    finger[PD_DIFF_WEIB_A], _ = calc_weibull_params(pd_diff)

    finger[TD_MEDIAN] = df[data.TIMEDIFF].median()

    finger[CORR_PD_DIFF_TO_PD] = _correlate_pd_and_pd_diff(df[data.PD][:-1], pd_diff)
    finger[CORR_NEXT_PD_TO_PD] = _correlate_pd_and_next_pd(
        df[data.PD][:-1], df[data.PD][1:].reset_index(drop=True)
    )

    return finger


def build_set(measurements: list, fingerprint: Callable = tu_graz) -> pd.DataFrame:
    fingers = pd.DataFrame([fingerprint(measurement) for measurement in measurements])

    defects = [measurement[data.CLASS][0] for measurement in measurements]
    fingers[data.CLASS] = pd.Series(defects, dtype="category")

    return fingers
