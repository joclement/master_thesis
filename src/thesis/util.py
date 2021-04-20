from pathlib import Path
from typing import Optional

from joblib import Memory
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import CACHE_DIR


def finish_plot(name: Optional[str], output_folder: Optional[Path], show: bool = False):
    if output_folder:
        plt.savefig(f"{output_folder}/{name}.svg")
    if show:
        plt.show()
    plt.close()


def flatten(t):
    return [item for sublist in t for item in sublist]


def to_dataTIME(time: pd.Timedelta) -> int:
    return int(time.value / 10 ** 6)


class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print(pd.DataFrame(X).head())
        print(X.shape)
        return X

    def fit(self, X, y=None, **fit_params):
        return self


def get_memory():
    return Memory(location=CACHE_DIR, verbose=0)
