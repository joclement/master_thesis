from typing import Tuple, Union

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
    finger[TD_WEIB_A], finger[TD_WEIB_B] = calc_weibull_params(
        df[data.TIMEDIFF][1:]
    )

    finger[PDS_PER_SEC] = len(df[data.TIME]) / df[data.TIME].max()

    return finger


def build_set(measurements: list) -> pd.DataFrame:
    fingers = pd.DataFrame([tu_graz(measurement) for measurement in measurements])

    defects = [measurement[data.CLASS][0] for measurement in measurements]
    fingers[data.CLASS] = pd.Series(defects, dtype="category")

    return fingers
