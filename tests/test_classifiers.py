import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import check_estimator

from thesis import classifiers, data


@pytest.fixture
def X_and_y(measurements):
    y = data.get_defects(measurements)
    y.append(y[0])
    X = pd.DataFrame({"value": list(range(len(y)))})
    return X, y


def test_LukasMeanDistance_checks_sklearn():
    check_estimator(classifiers.LukasMeanDist())


def test_DummyClassifier(X_and_y):
    X, y = X_and_y
    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(X, y)
    assert np.array_equal(dummy_classifier.predict(X), [y[0]] * len(y))
