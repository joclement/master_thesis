from typing import List, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC

from . import classifiers, prepared_data


def get_model_with_data(
    measurements: List[pd.DataFrame], model_name: str, model_config: dict
) -> Tuple[Union[Pipeline, BaseEstimator], pd.DataFrame]:
    parts = model_name.split("-")
    assert len(parts) == 2
    classifier_id = parts[0]
    data_id = parts[1]

    get_classifier = globals()[classifier_id]
    if "classifier_hyperparameters" in model_config:
        classifier = get_classifier(**model_config["classifier_hyperparameters"])
    else:
        classifier = get_classifier()

    get_input_data = getattr(prepared_data, data_id)
    input_data, transformer = get_input_data(measurements, **model_config)

    if transformer:
        pipeline = make_pipeline(transformer, classifier)
    else:
        pipeline = classifier
    return pipeline, input_data


def dt(**classifier_config: dict) -> BaseEstimator:
    return DecisionTreeClassifier()


def knn(**classifier_config: dict) -> BaseEstimator:
    return KNeighborsClassifier(**classifier_config)


def knn_dtw(**classifier_config: dict) -> BaseEstimator:
    return KNeighborsTimeSeriesClassifier(**classifier_config)


def mlp(**classifier_config: dict) -> BaseEstimator:
    return MLPClassifier(**classifier_config)


def ott_algo(**classifier_config) -> BaseEstimator:
    return classifiers.LukasMeanDist()


def svm(**classifier_config: dict) -> BaseEstimator:
    return SVC(**classifier_config)


def svm_dtw(**classifier_config: dict) -> BaseEstimator:
    return TimeSeriesSVC()
