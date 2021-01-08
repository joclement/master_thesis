import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


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
        X, y = check_X_y(X, y)
        mean_per_defect = _calc_mean_per_defect(X, y, list(unique_labels(y)))
        y_mean = list(mean_per_defect.keys())
        X_mean = pd.DataFrame(mean_per_defect.values())
        super().fit(X_mean, y_mean)
        return self

    def _more_tags(self):
        return {"multioutput": False}
