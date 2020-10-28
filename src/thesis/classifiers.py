import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


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


class LukasMeanDist(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = list(unique_labels(y))
        mean_per_defect = _calc_mean_per_defect(X, y, self.classes_)
        y_mean = list(mean_per_defect.keys())
        assert y_mean == self.classes_
        X_mean = pd.DataFrame(mean_per_defect.values())
        self.k_nn_ = KNeighborsClassifier(n_neighbors=1).fit(X_mean, y_mean)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        return self.k_nn_.predict(X)

    def get_params(self, deep=True):
        return dict()

    def set_params(self, **parameters):
        return self

    def _more_tags(self):
        return {"X_types": ["2darray", "1dlabels"]}
