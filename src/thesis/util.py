import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def finish_plot(name: str, output_folder, show: bool = False):
    if output_folder:
        plt.savefig(f"{output_folder}/{name}.svg")
    if show:
        plt.show()
    plt.close()


def to_dataTIME(time: pd.Timedelta) -> int:
    return int(time.value / 10 ** 6)


class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print(pd.DataFrame(X).head())
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
