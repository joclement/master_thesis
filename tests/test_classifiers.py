from sklearn.utils.estimator_checks import check_estimator

from thesis import classifiers


def test_LukasMeanDistance_checks_sklearn():
    check_estimator(classifiers.LukasMeanDist())
