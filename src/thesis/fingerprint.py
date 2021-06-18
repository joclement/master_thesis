from enum import Enum
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Set, Tuple, Union


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


PD_ID = "A"
PD_DIFF_ID = "\u0394A"
TD_ID = "\u0394t"
CORR_ID = "Corr"


class Group(Enum):
    pd = PD_ID
    pd_diff = PD_DIFF_ID
    td = TD_ID
    corr = CORR_ID

    def __str__(self):
        return "%s" % self.value


# @note: parameter in TU Graz fingerprint
PD_VAR = f"{PD_ID}-Var"
PD_SKEW = f"{PD_ID}-Skew"
PD_KURT = f"{PD_ID}-Kurt"
PD_WEIB_A = f"{PD_ID}-Weib-\u03B1"
PD_WEIB_B = f"{PD_ID} Weib-\u03B2"

PD_DIFF_WEIB_B = f"{PD_DIFF_ID}-Weib-\u03B2"

TD_MAX = f"{TD_ID}-Max"
TD_MEAN = f"{TD_ID}-Mean"
TD_MIN = f"{TD_ID}-Min"
TD_VAR = f"{TD_ID}-Var"
TD_SKEW = f"{TD_ID}-Skew"
TD_KURT = f"{TD_ID}-Kurt"
TDIFF_NORM_WEIB_A = f"{TD_ID}-Norm-Weib-\u03B1"
TDIFF_NORM_WEIB_B = f"{TD_ID}-Norm-Weib-\u03B2"


# @note: parameter in Lukas fingerprint
PDS_PER_SEC = "PDs/Sec"

PD_MEAN = f"{PD_ID}-Mean"
PD_MAX = f"{PD_ID}-Max"
PD_CV = f"{PD_ID}-Std/Mean"
PD_SUM = f"{PD_ID}-Sum"

PD_DIFF_MEAN = f"{PD_DIFF_ID} Mean"
PD_DIFF_SKEW = f"{PD_DIFF_ID} Skewness"
PD_DIFF_VAR = f"{PD_DIFF_ID} Variance"
PD_DIFF_KURT = f"{PD_DIFF_ID} Kurtosis"
PD_DIFF_WEIB_A = f"{PD_DIFF_ID} Weibull A"

TD_MEDIAN = f"{TD_ID}-Median"

CORR_PD_DIFF_TO_PD_BINS = f"{CORR_ID}-{PD_DIFF_ID}-{PD_ID}-Bins"
CORR_NEXT_PD_TO_PD_BINS = f"Auto-{CORR_ID}-{PD_ID}-Bins"
CORR_NEXT_PD_TO_PD = f"Auto-{CORR_ID}-{PD_ID}-1"
CORR_PD_DIFF_TO_PD = f"{CORR_ID}-{PD_DIFF_ID}-{PD_ID}"
CORR_PD_TO_TD = f"{CORR_ID}-{PD_ID}-{TD_ID}"

# @note: parameter in own fingerprint
POLARITY = "+DC/-DC"

PD_MIN = f"{PD_ID} min"
PD_MEDIAN = f"{PD_ID} Median"
PD_STD = f"{PD_ID} std"
PD_VAR = f"{PD_ID}-Var"
PD_NUM_PEAKS_50 = f"{PD_ID}-Num-peaks-50"
PD_NUM_PEAKS_10 = f"{PD_ID}-Num-peaks-10"
PD_NUM_PEAKS_5 = f"{PD_ID}-Num-peaks-5"
PD_NUM_PEAKS_100 = f"{PD_ID}-Num-peaks-100"
PD_RATIO = f"{PD_ID} ratio"
PD_PERC_REOCCUR = f"{PD_ID} percentage reocurring"
PD_COUNT_ABOVE_MEAN = f"{PD_ID}-Num->-mean"
PD_COUNT_BELOW_MEAN = f"{PD_ID}-Num-<-mean"
PD_CHANGE_QUANTILES = f"{PD_ID}-change-quantiles"
PD_NORM_WEIB_A = f"{PD_ID}-norm-Weib-\u03B1"
PD_NORM_WEIB_B = f"{PD_ID}-norm-Weib-\u03B2"
PD_LONGEST_STRIKE_BELOW_MEAN = f"{PD_ID}-max-strike-<-mean"

TD_CHANGE_QUANTILES = f"{TD_ID}-Change-quantiles"
TD_SUM = f"{TD_ID}-Sum"

CORR_2ND_NEXT_PD_TO_PD = f"Auto-{CORR_ID}-{PD_ID}-2"
CORR_3RD_NEXT_PD_TO_PD = f"Auto-{CORR_ID}-{PD_ID}-3"
CORR_5TH_NEXT_PD_TO_PD = f"Auto-{CORR_ID}-{PD_ID}-5"
CORR_10TH_NEXT_PD_TO_PD = f"Auto-{CORR_ID}-{PD_ID}-10"

AUTOCORR_NEXT_TD = f"Auto-{CORR_ID}-{TD_ID}-1"
AUTOCORR_2ND_NEXT_TD = f"Auto-{CORR_ID}-{TD_ID}-2"
AUTOCORR_3RD_NEXT_TD = f"Auto-{CORR_ID}-{TD_ID}-3"
AUTOCORR_5TH_NEXT_TD = f"Auto-{CORR_ID}-{TD_ID}-5"
AUTOCORR_10TH_NEXT_TD = f"Auto-{CORR_ID}-{TD_ID}-10"

# @note: further parameters
PD_BY_TD_WEIB_A = f"{PD_ID}/{TD_ID}-Weib-\u03B1"
PD_BY_TD_WEIB_B = f"{PD_ID}/{TD_ID}-Weibu-\u03B2"


def get_parameter_group(df: pd.DataFrame, group: Group) -> pd.DataFrame:
    wanted_columns = [column for column in df.columns if group.value in column]
    return df[wanted_columns].copy()


def calc_weibull_params(data: Union[list, pd.Series]) -> Tuple[float, float]:
    weibull_b_shape, _, weibull_a_scale = stats.weibull_min.fit(data, floc=0.0)
    return weibull_a_scale, weibull_b_shape


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


def make_distribution(values: pd.Series) -> pd.Series:
    distribution = values.sort_values()
    return distribution / distribution.max()


class Features:
    def __del__(self):
        if self.timing_filepath is not None:
            if not self.timing_filepath.exists():
                idx = pd.MultiIndex.from_tuples(
                    [(self.df_attrs[PATH], self.df_attrs[PART])],
                    name=self.get_index_col(),
                )
                self.time_df = pd.DataFrame(
                    data=[self.times.values()], index=idx, columns=self.times.keys()
                )
                Features.first_run = False
            else:
                self.time_df.loc[
                    (self.df_attrs[PATH], self.df_attrs[PART]), :
                ] = pd.Series(data=self.times.values(), index=self.times.keys())
            self.time_df.to_csv(self.timing_filepath)

    def get_index_col(self) -> List[str]:
        index_col = [PATH]
        if PART in self.df_attrs:
            index_col.append(PART)
        return index_col

    def __init__(self, df_attrs: dict, timing_filepath):
        self.timing_filepath = timing_filepath
        self.features: Dict[str, float] = {}
        self.times: Dict[str, float] = {}
        if self.timing_filepath is not None:
            self.timing_filepath = Path(timing_filepath)
            self.df_attrs = df_attrs
            if self.timing_filepath.exists():
                self.time_df = pd.read_csv(
                    self.timing_filepath, header=0, index_col=self.get_index_col()
                )

    def add(self, feature_id: Union[Tuple[str, str], str], function: Callable):
        start = time.process_time()
        result = function()
        duration = time.process_time() - start
        if isinstance(feature_id, str):
            self.features[feature_id] = result
            count_id = feature_id
        else:
            for id, val in zip(feature_id, result):
                self.features[id] = val
            count_id = "+".join(feature_id)
        self.times[count_id] = duration


@memory.cache
def extract_features(df: pd.DataFrame, timing_filepath: str = None):
    pd_diff = df[PD].diff()[1:].abs().reset_index(drop=True)

    features = Features(df.attrs, timing_filepath)
    features.add(
        CORR_NEXT_PD_TO_PD_BINS,
        lambda: _correlate_with_bins(df[PD][:-1], df[PD][1:].reset_index(drop=True)),
    ),
    features.add(
        CORR_PD_DIFF_TO_PD_BINS,
        lambda: _correlate_with_bins(df[PD][1:].reset_index(drop=True), pd_diff),
    ),
    features.add(CORR_NEXT_PD_TO_PD, lambda: autocorrelate(df[PD], 1)),
    features.add(CORR_2ND_NEXT_PD_TO_PD, lambda: autocorrelate(df[PD], 2)),
    features.add(CORR_3RD_NEXT_PD_TO_PD, lambda: autocorrelate(df[PD], 3)),
    features.add(CORR_5TH_NEXT_PD_TO_PD, lambda: autocorrelate(df[PD], 5)),
    features.add(CORR_10TH_NEXT_PD_TO_PD, lambda: autocorrelate(df[PD], 10)),
    features.add(
        CORR_PD_DIFF_TO_PD,
        lambda: stats.pearsonr(df[PD][1:].reset_index(drop=True), pd_diff)[0],
    ),
    features.add(CORR_PD_TO_TD, lambda: stats.pearsonr(df[PD], df[TIME_DIFF])[0]),
    features.add(AUTOCORR_NEXT_TD, lambda: autocorrelate(df[TIME_DIFF], 1)),
    features.add(AUTOCORR_2ND_NEXT_TD, lambda: autocorrelate(df[TIME_DIFF], 2)),
    features.add(AUTOCORR_3RD_NEXT_TD, lambda: autocorrelate(df[TIME_DIFF], 3)),
    features.add(AUTOCORR_5TH_NEXT_TD, lambda: autocorrelate(df[TIME_DIFF], 5)),
    features.add(AUTOCORR_10TH_NEXT_TD, lambda: autocorrelate(df[TIME_DIFF], 10)),
    features.add(PDS_PER_SEC, lambda: len(df.index) / (df[TIME_DIFF].sum() / 1000)),
    features.add(
        PD_CHANGE_QUANTILES, lambda: change_quantiles(df[PD], 0.0, 0.7, True, "mean")
    ),
    features.add(PD_COUNT_ABOVE_MEAN, lambda: count_above_mean(df[PD]) / len(df.index)),
    features.add(PD_COUNT_BELOW_MEAN, lambda: count_below_mean(df[PD]) / len(df.index)),
    features.add(PD_CV, lambda: df[PD].std() / df[PD].mean()),
    features.add(PD_DIFF_KURT, lambda: pd_diff.kurt()),
    features.add(PD_DIFF_MEAN, lambda: pd_diff.mean()),
    features.add(PD_DIFF_SKEW, lambda: pd_diff.skew()),
    features.add(PD_DIFF_VAR, lambda: pd_diff.var()),
    features.add(PD_KURT, lambda: df[PD].kurt()),
    features.add(PD_MAX, lambda: df[PD].max()),
    features.add(PD_MEAN, lambda: df[PD].mean()),
    features.add(PD_MEDIAN, lambda: df[PD].median()),
    features.add(PD_MIN, lambda: df[PD].min()),
    features.add(PD_NUM_PEAKS_10, lambda: number_peaks(df[PD], 10)),
    features.add(PD_NUM_PEAKS_5, lambda: number_peaks(df[PD], 5)),
    features.add(PD_NUM_PEAKS_50, lambda: number_peaks(df[PD], 50)),
    features.add(PD_NUM_PEAKS_100, lambda: number_peaks(df[PD], 100)),
    features.add(
        PD_PERC_REOCCUR,
        lambda: percentage_of_reoccurring_datapoints_to_all_datapoints(df[PD]),
    ),
    features.add(PD_RATIO, lambda: ratio_value_number_to_time_series_length(df[PD])),
    features.add(PD_SKEW, lambda: df[PD].skew()),
    features.add(PD_STD, lambda: df[PD].std()),
    features.add(PD_SUM, lambda: df[PD].sum()),
    features.add(PD_VAR, lambda: df[PD].var()),
    features.add(
        TD_CHANGE_QUANTILES,
        lambda: change_quantiles(df[TIME_DIFF], 0.0, 0.3, True, "var"),
    ),
    features.add(TD_KURT, lambda: df[TIME_DIFF].kurt()),
    features.add(
        PD_LONGEST_STRIKE_BELOW_MEAN,
        lambda: longest_strike_below_mean(df[PD]) / len(df.index),
    ),
    features.add(TD_MAX, lambda: df[TIME_DIFF].max()),
    features.add(TD_MEAN, lambda: df[TIME_DIFF].mean()),
    features.add(TD_MEDIAN, lambda: df[TIME_DIFF].median()),
    features.add(TD_MIN, lambda: df[TIME_DIFF].min()),
    features.add(TD_SKEW, lambda: df[TIME_DIFF].skew()),
    features.add(TD_SUM, lambda: df[TIME_DIFF].sum()),
    features.add(TD_VAR, lambda: df[TIME_DIFF].var()),
    features.add(POLARITY, lambda: df.attrs[VOLTAGE_SIGN]),

    features.add((PD_DIFF_WEIB_A, PD_DIFF_WEIB_B), lambda: calc_weibull_params(pd_diff))
    features.add(
        (PD_NORM_WEIB_A, PD_NORM_WEIB_B),
        lambda: calc_weibull_params(make_distribution(df[PD])),
    )
    features.add(
        (TDIFF_NORM_WEIB_A, TDIFF_NORM_WEIB_B),
        lambda: calc_weibull_params(make_distribution(df[TIME_DIFF])),
    )
    features.add(
        (PD_BY_TD_WEIB_A, PD_BY_TD_WEIB_B),
        lambda: calc_weibull_params(make_distribution(df[PD] / df[TIME_DIFF])),
    )
    features.add((PD_WEIB_A, PD_WEIB_B), lambda: calc_weibull_params(df[PD]))

    extracted_features = pd.DataFrame(
        data=features.features,
        columns=features.features.keys(),
        index=[get_X_index(df)],
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
    feature(AUTOCORR_5TH_NEXT_TD),
    feature(AUTOCORR_NEXT_TD),
    feature(CORR_2ND_NEXT_PD_TO_PD),
    feature(CORR_3RD_NEXT_PD_TO_PD),
    feature(CORR_NEXT_PD_TO_PD),
    feature(CORR_PD_DIFF_TO_PD),
    feature(CORR_PD_TO_TD),
    feature(PD_BY_TD_WEIB_A),
    feature(PD_BY_TD_WEIB_B),
    feature(PD_COUNT_ABOVE_MEAN),
    feature(PD_NORM_WEIB_A),
    feature(PD_NORM_WEIB_B),
    feature(PD_NUM_PEAKS_5),
    feature(PDS_PER_SEC),
    feature(PD_VAR),
    feature(POLARITY),
    feature(TDIFF_NORM_WEIB_A),
    feature(TDIFF_NORM_WEIB_B),
    feature(TD_CHANGE_QUANTILES),
    feature(TD_KURT),
    feature(PD_LONGEST_STRIKE_BELOW_MEAN),
    feature(TD_MEDIAN),
    feature(TD_SKEW),
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
