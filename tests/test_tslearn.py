import random

import numpy as np
import pytest
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset


@pytest.fixture
def data():
    data = [
        np.random.default_rng().uniform(0, high=2.3, size=(random.randint(4, 46), 12))
        for i in range(50)
    ]
    return data


def test_TimeSeriesScalerMinMax_passes_sklearn():
    check_estimator(TimeSeriesScalerMinMax())


def test_to_timeseries_dataset(data):
    X = to_time_series_dataset(data)
    assert np.array_equal(to_time_series_dataset(X), X, equal_nan=True)
    assert np.array_equal(
        to_time_series_dataset(to_time_series_dataset(X)), X, equal_nan=True
    )


def test_variable_lengths_classification(data):
    X = to_time_series_dataset(data)
    X = TimeSeriesScalerMinMax().fit_transform(X)
    assert X.shape[0] == 50
    assert X.shape[2] == 12
    assert not np.isinf(X).any()
    assert np.isnan(X).any()

    y = np.random.randint(0, 3, 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    knn = KNeighborsTimeSeriesClassifier()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    assert 0 <= metrics.accuracy_score(predictions, y_test) <= 1
