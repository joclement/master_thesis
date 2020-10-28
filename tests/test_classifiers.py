from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import check_estimator

from thesis import classifiers


def test_kNN_sparse_in__get_tags():
    k_nn = KNeighborsClassifier(n_neighbors=1)
    assert k_nn._get_tags()["allow_nan"] is False
    assert "sparse" not in k_nn._get_tags()["X_types"]
    assert "2darray" in k_nn._get_tags()["X_types"]
    assert "1dlabels" not in k_nn._get_tags()["X_types"]
    check_estimator(k_nn)


def test_LukasMeanDistance_checks_sklearn():
    check_estimator(classifiers.LukasMeanDist())
