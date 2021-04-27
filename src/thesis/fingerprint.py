from enum import Enum
import math
from typing import Any, Callable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from tsfresh.feature_extraction.feature_calculators import (
    change_quantiles,
    count_above_mean,
    count_below_mean,
    longest_strike_below_mean,
    number_peaks,
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    ratio_value_number_to_time_series_length,
)

from .constants import PART
from .data import CLASS, get_defects, PATH, PD, TIME_DIFF, VOLTAGE_SIGN
from .util import get_memory

memory = get_memory()

PD_ID = "PD-Value"
PD_DIFF_ID = "PD-Diff"
TD_ID = "TimeDiff"
CORR_ID = "Correlate"


class Group(Enum):
    pd = PD_ID
    pd_diff = PD_DIFF_ID
    td = TD_ID
    corr = CORR_ID

    def __str__(self):
        return "%s" % self.value


# @note: parameter in TU Graz fingerprint
PD_VAR = f"{PD_ID} Variance"
PD_SKEW = f"{PD_ID} Skewness"
PD_KURT = f"{PD_ID} Kurtosis"
PD_WEIB_A = f"{PD_ID} Weibull A"
PD_WEIB_B = f"{PD_ID} Weibull B"

PD_DIFF_WEIB_B = f"{PD_DIFF_ID} Weibull B"

TD_MAX = f"{TD_ID} Max"
TD_MEAN = f"{TD_ID} Mean"
TD_MIN = f"{TD_ID} Min"
TD_VAR = f"{TD_ID} Variance"
TD_SKEW = f"{TD_ID} Skewness"
TD_KURT = f"{TD_ID} Kurtosis"
TDIFF_NORM_WEIB_A = f"{TD_ID} Sorted Normed Weibull A"
TDIFF_NORM_WEIB_B = f"{TD_ID} Sorted Normed Weibull B"


# @note: parameter in Lukas fingerprint
PDS_PER_SEC = "Number of PDs/sec"

PD_MEAN = f"{PD_ID} Mean"
PD_MAX = f"{PD_ID} Max"
PD_CV = f"{PD_ID} std/mean"
PD_SUM = f"{PD_ID} Sum"

PD_DIFF_MEAN = f"{PD_DIFF_ID} Mean"
PD_DIFF_SKEW = f"{PD_DIFF_ID} Skewness"
PD_DIFF_VAR = f"{PD_DIFF_ID} Variance"
PD_DIFF_KURT = f"{PD_DIFF_ID} Kurtosis"
PD_DIFF_WEIB_A = f"{PD_DIFF_ID} Weibull A"

TD_MEDIAN = f"{TD_ID} Median"

CORR_PD_DIFF_TO_PD_BINS = f"{CORR_ID} {PD_DIFF_ID} - PD Bins"
CORR_NEXT_PD_TO_PD_BINS = f"{CORR_ID} Next PD - PD Bins"
CORR_NEXT_PD_TO_PD = f"{CORR_ID} Next PD - PD"
CORR_PD_DIFF_TO_PD = f"{CORR_ID} {PD_DIFF_ID} - PD"
CORR_PD_DIFF_TO_TD = f"{CORR_ID} PD - {PD_DIFF_ID}"
CORR_PD_TO_TD = f"{CORR_ID} PD - {TD_ID}"

# @note: parameter in own fingerprint
POLARITY = "+DC/-DC"

PD_MIN = f"{PD_ID} min"
PD_MEDIAN = f"{PD_ID} Median"
PD_STD = f"{PD_ID} std"
PD_VAR = f"{PD_ID} var"
SIZE = f"{PD_ID} Num"
PD_NUM_PEAKS_50 = f"{PD_ID} num peaks 50"
PD_NUM_PEAKS_10 = f"{PD_ID} num peaks 10"
PD_NUM_PEAKS_5 = f"{PD_ID} num peaks 5"
PD_NUM_PEAKS_100 = f"{PD_ID} num peaks 100"
PD_RATIO = f"{PD_ID} ratio"
PD_PERC_REOCCUR = f"{PD_ID} percentage reocurring"
PD_COUNT_ABOVE_MEAN = f"{PD_ID} count above mean"
PD_COUNT_BELOW_MEAN = f"{PD_ID} count below mean"
PD_CHANGE_QUANTILES = f"{PD_ID} ChangeQuantiles"
PD_NORM_WEIB_A = f"{PD_ID} Weibull normed sorted A"
PD_NORM_WEIB_B = f"{PD_ID} Weibull normed sorted B"

TD_LONGEST_STRIKE_BELOW_MEAN = f"{TD_ID} longest strike below mean"
TD_CHANGE_QUANTILES = f"{TD_ID} ChangeQuantiles"
TD_SUM = f"{TD_ID} Sum"

CORR_2ND_NEXT_PD_TO_PD = f"{CORR_ID} 2nd Next PD - PD"
CORR_3RD_NEXT_PD_TO_PD = f"{CORR_ID} 3rd Next PD - PD"
CORR_5TH_NEXT_PD_TO_PD = f"{CORR_ID} 5th Next PD - PD"
CORR_10TH_NEXT_PD_TO_PD = f"{CORR_ID} 10th Next PD - PD"

AUTOCORR_NEXT_TD = f"{CORR_ID} Auto Next TD"
AUTOCORR_2ND_NEXT_TD = f"{CORR_ID} Auto 2nd Next TD"
AUTOCORR_3RD_NEXT_TD = f"{CORR_ID} Auto 3rd Next TD"
AUTOCORR_5TH_NEXT_TD = f"{CORR_ID} Auto 5th Next TD"
AUTOCORR_10TH_NEXT_TD = f"{CORR_ID} Auto 10th Next TD"

# @note: further parameters
PD_BY_TD_WEIB_A = f"{PD_ID} / {TD_ID} Weibull A"
PD_BY_TD_WEIB_B = f"{PD_ID} / {TD_ID} Weibull B"


def get_parameter_group(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    wanted_columns = [column for column in df.columns if group.value in column]
    return df[wanted_columns].copy()


@memory.cache
def calc_weibull_params(data: Union[list, pd.Series]) -> Tuple[float, float]:
    weibull_b, _, weibull_a = stats.weibull_min.fit(data, floc=0.0)
    return weibull_a, weibull_b


CATEGORICAL_FEATURES = {POLARITY}


def get_categorical_features(feature_union: FeatureUnion) -> Set[str]:
    return set.intersection(CATEGORICAL_FEATURES, set(get_feature_names(feature_union)))


def autocorrelate(values: pd.Series, lag: int) -> float:
    if len(values) < 2 * lag:
        return 0.0
    corr_coef = stats.pearsonr(values[:-lag], values[lag:])[0]
    if corr_coef is np.nan:
        return 0.0
    return corr_coef


@memory.cache
def extract_features(df: pd.DataFrame):
    pd_diff = df[PD].diff()[1:].abs().reset_index(drop=True)
    features = {
        CORR_NEXT_PD_TO_PD_BINS: _correlate_with_bins(
            df[PD][:-1], df[PD][1:].reset_index(drop=True)
        ),
        CORR_PD_DIFF_TO_PD_BINS: _correlate_with_bins(
            df[PD][1:].reset_index(drop=True), pd_diff
        ),
        CORR_NEXT_PD_TO_PD: autocorrelate(df[PD], 1),
        CORR_2ND_NEXT_PD_TO_PD: autocorrelate(df[PD], 2),
        CORR_3RD_NEXT_PD_TO_PD: autocorrelate(df[PD], 3),
        CORR_5TH_NEXT_PD_TO_PD: autocorrelate(df[PD], 5),
        CORR_10TH_NEXT_PD_TO_PD: autocorrelate(df[PD], 10),
        CORR_PD_DIFF_TO_PD: stats.pearsonr(df[PD][1:].reset_index(drop=True), pd_diff)[
            0
        ],
        CORR_PD_TO_TD: stats.pearsonr(df[PD], df[TIME_DIFF])[0],
        AUTOCORR_NEXT_TD: autocorrelate(df[TIME_DIFF], 1),
        AUTOCORR_2ND_NEXT_TD: autocorrelate(df[TIME_DIFF], 2),
        AUTOCORR_3RD_NEXT_TD: autocorrelate(df[TIME_DIFF], 3),
        AUTOCORR_5TH_NEXT_TD: autocorrelate(df[TIME_DIFF], 5),
        AUTOCORR_10TH_NEXT_TD: autocorrelate(df[TIME_DIFF], 10),
        PDS_PER_SEC: len(df[TIME_DIFF]) / (df[TIME_DIFF].sum() / 1000),
        PD_CHANGE_QUANTILES: change_quantiles(df[PD], 0.0, 0.7, True, "mean"),
        PD_COUNT_ABOVE_MEAN: count_above_mean(df[PD]),
        PD_COUNT_BELOW_MEAN: count_below_mean(df[PD]),
        PD_CV: df[PD].std() / df[PD].mean(),
        PD_DIFF_KURT: pd_diff.kurt(),
        PD_DIFF_MEAN: pd_diff.mean(),
        PD_DIFF_SKEW: pd_diff.skew(),
        PD_DIFF_VAR: pd_diff.var(),
        PD_KURT: df[PD].kurt(),
        PD_MAX: df[PD].max(),
        PD_MEAN: df[PD].mean(),
        PD_MEDIAN: df[PD].median(),
        PD_MIN: df[PD].min(),
        PD_NUM_PEAKS_10: number_peaks(df[PD], 10),
        PD_NUM_PEAKS_5: number_peaks(df[PD], 5),
        PD_NUM_PEAKS_50: number_peaks(df[PD], 50),
        PD_NUM_PEAKS_100: number_peaks(df[PD], 100),
        PD_PERC_REOCCUR: percentage_of_reoccurring_datapoints_to_all_datapoints(df[PD]),
        PD_RATIO: ratio_value_number_to_time_series_length(df[PD]),
        PD_SKEW: df[PD].skew(),
        PD_STD: df[PD].std(),
        PD_SUM: df[PD].sum(),
        PD_VAR: df[PD].var(),
        SIZE: len(df.index),
        TD_CHANGE_QUANTILES: change_quantiles(df[TIME_DIFF], 0.0, 0.3, True, "var"),
        TD_KURT: df[TIME_DIFF].kurt(),
        TD_LONGEST_STRIKE_BELOW_MEAN: longest_strike_below_mean(df[PD]),
        TD_MAX: df[TIME_DIFF].max(),
        TD_MEAN: df[TIME_DIFF].mean(),
        TD_MEDIAN: df[TIME_DIFF].median(),
        TD_MIN: df[TIME_DIFF].min(),
        TD_SKEW: df[TIME_DIFF].skew(),
        TD_SUM: df[TIME_DIFF].sum(),
        TD_VAR: df[TIME_DIFF].var(),
        POLARITY: df.attrs[VOLTAGE_SIGN],
    }

    (
        features[PD_DIFF_WEIB_A],
        features[PD_DIFF_WEIB_B],
    ) = calc_weibull_params(pd_diff)
    (
        features[PD_NORM_WEIB_A],
        features[PD_NORM_WEIB_B],
    ) = calc_weibull_params(df[PD].sort_values() / df[PD].max())
    (
        features[TDIFF_NORM_WEIB_A],
        features[TDIFF_NORM_WEIB_B],
    ) = calc_weibull_params(df[TIME_DIFF].cumsum() / df[TIME_DIFF].sum())

    pd_by_timediff = (df[PD] / df[TIME_DIFF]).sort_values()
    pd_by_timediff /= pd_by_timediff.max()
    features[PD_BY_TD_WEIB_A], features[PD_BY_TD_WEIB_B] = calc_weibull_params(
        pd_by_timediff
    )

    features[PD_WEIB_A], features[PD_WEIB_B] = calc_weibull_params(df[PD])

    extracted_features = pd.DataFrame(
        data=features, columns=features.keys(), index=[get_X_index(df)]
    )
    if math.isnan(extracted_features[PD_DIFF_KURT]):
        extracted_features[PD_DIFF_KURT] = 0.0
    if (
        extracted_features.isnull().any().any()
        or extracted_features.isin([np.inf, -np.inf]).any().any()
    ):
        raise ValueError(f"Incorrect features: \n {extracted_features.to_string()}")
    return extracted_features


# Based on https://doi.org/10.5324/nordis.v0i26.3268
def tugraz_feature_union(**data_config) -> FeatureUnion:
    return FeatureUnion(
        [
            feature(PDS_PER_SEC),
            feature(PD_DIFF_WEIB_A),
            feature(PD_DIFF_WEIB_B),
            feature(PD_KURT),
            feature(PD_SKEW),
            feature(PD_VAR),
            feature(TDIFF_NORM_WEIB_A),
            feature(TDIFF_NORM_WEIB_B),
            feature(TD_KURT),
            feature(TD_MAX),
            feature(TD_MEAN),
            feature(TD_MIN),
            feature(TD_SKEW),
            feature(TD_VAR),
        ],
        **data_config,
    )


def tugraz(df: pd.DataFrame) -> pd.Series:
    return build_set([df], tugraz).iloc[0]


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


def ott_feature_union(**config):
    return FeatureUnion(
        [
            feature(PD_MEAN),
            feature(PD_CV),
            feature(PD_MAX),
            feature(PD_WEIB_A),
            feature(PD_WEIB_B),
            feature(PD_DIFF_MEAN),
            feature(PD_DIFF_SKEW),
            feature(PD_DIFF_KURT),
            feature(PD_DIFF_WEIB_A),
            feature(TD_MEDIAN),
            feature(CORR_PD_DIFF_TO_PD_BINS),
            feature(CORR_NEXT_PD_TO_PD_BINS),
        ]
    )


def ott(df: pd.DataFrame) -> pd.Series:
    return build_set([df], ott).iloc[0]


class Feature(TransformerMixin, BaseEstimator):
    def __init__(self, feature: str, **kwargs):
        self.feature = feature

    def fit(self, X: List[Any], y=None, **kwargs):
        self.n_features_in_ = 1
        return self

    def transform(self, X: pd.DataFrame, y=None, **kwargs):
        feature = X[self.feature].to_frame()
        if len(feature.index) == 0:
            raise ValueError("Non matching data.")
        return feature

    def get_feature_names(self):
        return self.feature

    def _more_tags(self):
        return {"no_validation": True, "requires_fit": False}


def feature(feature_id) -> Tuple[str, Feature]:
    return feature_id, Feature(feature_id)


EXTRA_OWN_FEATURES = [
    feature(PD_MAX),
    feature(PD_MEAN),
    feature(PD_MEDIAN),
    feature(PD_MIN),
    feature(PD_SUM),
]


def own_feature_union(**data_config) -> FeatureUnion:
    return FeatureUnion(RELOWN_FEATURES + EXTRA_OWN_FEATURES, **data_config)


RELOWN_FEATURES = [
    feature(AUTOCORR_10TH_NEXT_TD),
    feature(AUTOCORR_2ND_NEXT_TD),
    feature(AUTOCORR_3RD_NEXT_TD),
    feature(AUTOCORR_5TH_NEXT_TD),
    feature(AUTOCORR_NEXT_TD),
    feature(CORR_10TH_NEXT_PD_TO_PD),
    feature(CORR_2ND_NEXT_PD_TO_PD),
    feature(CORR_3RD_NEXT_PD_TO_PD),
    feature(CORR_5TH_NEXT_PD_TO_PD),
    feature(CORR_NEXT_PD_TO_PD),
    feature(CORR_PD_DIFF_TO_PD),
    feature(CORR_PD_TO_TD),
    feature(PD_BY_TD_WEIB_A),
    feature(PD_BY_TD_WEIB_B),
    feature(PD_CHANGE_QUANTILES),
    feature(PD_COUNT_ABOVE_MEAN),
    feature(PD_COUNT_BELOW_MEAN),
    feature(PD_NORM_WEIB_A),
    feature(PD_NORM_WEIB_B),
    feature(PD_NUM_PEAKS_10),
    feature(PD_NUM_PEAKS_5),
    feature(PD_VAR),
    feature(POLARITY),
    feature(SIZE),
    feature(TDIFF_NORM_WEIB_A),
    feature(TDIFF_NORM_WEIB_B),
    feature(TD_CHANGE_QUANTILES),
    feature(TD_KURT),
    feature(TD_LONGEST_STRIKE_BELOW_MEAN),
    feature(TD_MEDIAN),
    feature(TD_SKEW),
    feature(TD_SUM),
]


def relown_feature_union(**data_config) -> FeatureUnion:
    return FeatureUnion(RELOWN_FEATURES, **data_config)


def relown(df: pd.DataFrame) -> pd.Series:
    return build_set([df], relown).iloc[0]


def own(df: pd.DataFrame) -> pd.Series:
    return build_set([df], own).iloc[0]


def seqown(df: pd.DataFrame) -> pd.Series:
    own = [
        df[PD].mean(),
        df[PD].std(),
        df[PD].median(),
        df[PD].max(),
        df[PD].sum(),
        df[PD].var(),
        len(df.index),
        df[TIME_DIFF].skew(),
        df[TIME_DIFF].median(),
        number_peaks(df[TIME_DIFF], 3),
        df[TIME_DIFF].sum(),
    ]

    finger = pd.Series(data=own, dtype=float)

    if finger.isnull().any() or finger.isin([np.inf, -np.inf]).any():
        raise ValueError(f"Incorrect finger: \n {finger}")
    return finger


def get_X_index(df: pd.DataFrame) -> Union[str, Tuple[str, int]]:
    if PART in df.attrs:
        return df.attrs[PATH], df.attrs[PART]
    else:
        return df.attrs[PATH]


def get_feature_names(fingerTransformer: FeatureUnion) -> List[str]:
    return [name for name, _ in fingerTransformer.transformer_list]


def build_set(
    measurements: List[pd.DataFrame],
    fingerprint: Callable,
    add_class: bool = False,
) -> pd.DataFrame:
    FINGER_FEATURE_UNION_MAP = {
        own: own_feature_union,
        relown: relown_feature_union,
        tugraz: tugraz_feature_union,
        ott: ott_feature_union,
    }

    features = []
    for df in measurements:
        features.append(extract_features(df))
    features = pd.concat(features)

    fingerBuilder = FINGER_FEATURE_UNION_MAP[fingerprint]()
    fingers = fingerBuilder.transform(features)
    fingers = pd.DataFrame(data=fingers, columns=get_feature_names(fingerBuilder))
    if add_class:
        fingers[CLASS] = pd.Series(get_defects(measurements), dtype="category")
    return fingers
