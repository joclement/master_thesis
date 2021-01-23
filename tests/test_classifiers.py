import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.utils.estimator_checks import check_estimator

from thesis import classifiers, data


def test_LukasMeanDistance_checks_sklearn():
    check_estimator(classifiers.LukasMeanDist())


def test_DummyClassifier(measurements):
    y = data.get_defects(measurements)
    y.append(y[0])
    X = pd.DataFrame({"value": list(range(len(y)))})

    dummy_classifier = DummyClassifier(strategy="most_frequent")
    dummy_classifier.fit(X, y)
    assert np.array_equal(dummy_classifier.predict(X), [y[0]] * len(y))
