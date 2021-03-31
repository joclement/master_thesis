import math
import random
from typing import Dict, List, Tuple

from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

from .data import CLASS, VOLTAGE_SIGN, VoltageSign


def _calc_mean_per_defect(x_train, y_train, defects):
    fingers_per_defect = {defect: [] for defect in defects}
    for defect, finger in zip(y_train, x_train):
        fingers_per_defect[defect].append(finger)
    dataframe_per_defect = {
        defect: pd.DataFrame(fingers) for defect, fingers in fingers_per_defect.items()
    }
    mean_per_defect = dict()
    for defect, df in dataframe_per_defect.items():
        mean_per_defect[defect] = df.mean(axis=0, skipna=False)
    return mean_per_defect


class LukasMeanDist(KNeighborsClassifier):
    def __init__(self):
        super().__init__(n_neighbors=1)

    def fit(self, X, y):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        X, y = check_X_y(X, y)
        mean_per_defect = _calc_mean_per_defect(X, y, list(unique_labels(y)))
        y_mean = list(mean_per_defect.keys())
        X_mean = pd.DataFrame(mean_per_defect.values())
        super().fit(X_mean, y_mean)
        return self

    def _more_tags(self):
        return {"multioutput": False, "requires_y": True}


class MyKerasClassifier(KerasClassifier):
    def fit(self, X, y, **kwargs):
        super().set_params(**{"input_dim": X.shape[1]})
        self.history = super().fit(X, y, **kwargs)
        return self


def create_samples(
    measurements: List[pd.DataFrame], y, min_len
) -> Dict[Tuple[int, VoltageSign], List[pd.DataFrame]]:
    parts = {d: 0 for d in set(y)}
    for df in measurements:
        parts[df.attrs[CLASS]] += math.floor(len(df) / min_len)
    repeats = min(parts.values())
    samples: Dict[Tuple[int, VoltageSign], List[pd.DataFrame]] = {
        (d, v): [] for d in set(y) for v in VoltageSign
    }
    for df in measurements:
        i = 0
        pos = 0
        while i <= repeats and pos + min_len <= len(df):
            samples[(df.attrs[CLASS], df.attrs[VOLTAGE_SIGN])].append(
                df.loc[pos : pos + min_len, :].reset_index(drop=True)
            )
            pos += min_len
            i += 1
    return samples


def convert(measurements: List[pd.DataFrame], min_len):
    X = []
    ranges = []
    last = 0
    for df in measurements:
        parts = [
            df.iloc[pos : pos + min_len, :].reset_index(drop=True)
            for pos in range(0, len(df) - min_len + 1, min_len)
        ]
        assert len(parts) >= 1
        parts = parts[:30]
        ranges.append(range(last, last + len(parts)))
        last += len(parts)
        X.extend(parts)
    return to_time_series_dataset(X), ranges


MIN_LEN = 247


class UnderSampleKNN(KNeighborsTimeSeriesClassifier):
    def __init__(
        self,
        min_len,
        n_neighbors=5,
        weights="uniform",
        metric="dtw",
        metric_params=None,
        n_jobs=None,
        **kw_args
    ):
        self.min_len = min_len
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kw_args
        )

    def fit(self, measurements, y, **kwargs):
        samples = create_samples(measurements, y, self.min_len)
        self.min_samples = min(len(dfs) for dfs in samples.values())
        X = []
        y = []
        for k, v in samples.items():
            random.shuffle(v)
            X.extend(v[: self.min_samples])
            y.extend([k[0]] * self.min_samples)
        super().fit(to_time_series_dataset(X), y, **kwargs)
        return self

    def convert(self, measurements: List[pd.DataFrame]):
        return convert(measurements, self.min_len)

    def predict(self, measurements):
        X, ranges = self.convert(measurements)
        predictions = super().predict(X)
        return [stats.mode(predictions[r])[0] for r in ranges]

    def predict_proba(self, measurements):
        proba_predictions = []
        for df in measurements:
            X = self.convert([df])[0]
            proba_prediction = np.sum(super().predict_proba(X), axis=0) / len(X)
            proba_predictions.append(proba_prediction)
        return np.array(proba_predictions)

    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({"min_len": self.min_len})
        return params

    def set_params(self, **params):
        if "min_len" in params:
            self.min_len = params["min_len"]
            del params["min_len"]
        return super().set_params(**params)


class PolarityKNN(BaseEstimator, TransformerMixin):
    def __init__(self, min_len=0, neg_neighbors=1, pos_neighbors=1, **kwargs):
        self.min_len = min_len
        self.neg_neighbors = neg_neighbors
        self.pos_neighbors = pos_neighbors

    def fit(self, measurements, y, **kwargs):
        samples = create_samples(measurements, y, self.min_len)
        self.neg_min_samples = min(
            len(dfs) for key, dfs in samples.items() if key[1] == VoltageSign.negative
        )
        self.pos_min_samples = min(
            len(dfs) for key, dfs in samples.items() if key[1] == VoltageSign.positive
        )
        neg_X, neg_y = [], []
        pos_X, pos_y = [], []
        for key, v in samples.items():
            defect, voltage = key
            random.shuffle(v)
            if voltage == VoltageSign.positive:
                pos_X.extend(v[: self.pos_min_samples])
                pos_y.extend([defect] * self.pos_min_samples)
            elif voltage == VoltageSign.negative:
                neg_X.extend(v[: self.neg_min_samples])
                neg_y.extend([defect] * self.neg_min_samples)

        self.neg_knn = KNeighborsTimeSeriesClassifier(
            n_neighbors=self.neg_neighbors
        ).fit(to_time_series_dataset(neg_X), neg_y)
        self.pos_knn = KNeighborsTimeSeriesClassifier(
            n_neighbors=self.pos_neighbors
        ).fit(to_time_series_dataset(pos_X), pos_y)

        return self

    def convert(self, measurements: List[pd.DataFrame]):
        return convert(measurements, self.min_len)

    def predict(self, measurements):
        X, ranges = self.convert(measurements)
        predictions = super().predict(X)
        return [stats.mode(predictions[r])[0] for r in ranges]

    def predict_proba(self, measurements):
        proba_predictions = []
        for df in measurements:
            X = self.convert([df])[0]
            if df.attrs[VOLTAGE_SIGN] == VoltageSign.negative:
                proba_prediction = self.neg_knn.predict_proba(X)
            elif df.attrs[VOLTAGE_SIGN] == VoltageSign.positive:
                proba_prediction = self.pos_knn.predict_proba(X)
            proba_prediction = np.sum(proba_prediction, axis=0) / len(X)
            print(proba_prediction)
            proba_predictions.append(proba_prediction)
        return np.array(proba_predictions)
