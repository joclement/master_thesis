from typing import List, Tuple, Union

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC

from . import classifiers, prepared_data


def get_model_with_data(
    measurements: List[pd.DataFrame],
    y: Union[pd.Series, np.array],
    model_name: str,
    model_config: dict,
) -> Tuple[Union[Pipeline, BaseEstimator], pd.DataFrame]:
    parts = model_name.split("-")
    assert len(parts) == 2
    classifier_id = parts[0]
    data_id = parts[1]

    get_input_data = getattr(prepared_data, data_id)
    input_data, transformer = get_input_data(measurements, **model_config)

    if "classifier_hyperparameters" in model_config:
        classifier_config = model_config["classifier_hyperparameters"]
    else:
        classifier_config = {}
    classifier = get_classifier(
        classifier_id,
        input_data,
        set(y),
        **classifier_config,
    )

    if transformer:
        pipeline = make_pipeline(transformer, classifier)
    else:
        pipeline = make_pipeline(classifier)
    return pipeline, input_data


def get_classifier(
    classifier_id: str,
    input_data: pd.DataFrame,
    defects: set,
    **classifier_config: dict
) -> BaseEstimator:
    if classifier_id == "dt":
        return DecisionTreeClassifier()
    if classifier_id == "knn":
        return KNeighborsClassifier(**classifier_config)
    if classifier_id == "knn_dtw":
        return KNeighborsTimeSeriesClassifier(**classifier_config)
    if classifier_id == "mlp":
        return get_mlp(input_data, defects, **classifier_config)
    if classifier_id == "ott_algo":
        return classifiers.LukasMeanDist()
    if classifier_id == "svm":
        return SVC(**classifier_config)
    if classifier_id == "svm_dtw":
        return TimeSeriesSVC()


def build_mlp(
    input_dim: int, output_dim: int, hidden_layer_sizes: List[int], optimizer: str
) -> KerasClassifier:
    model = Sequential()
    neurons_per_layer = hidden_layer_sizes
    model.add(keras.Input(shape=(input_dim,)))
    model.add(Dense(neurons_per_layer[0], activation="relu"))
    model.add(
        Dense(output_dim, activation="softmax"),
    )
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def get_mlp(
    input_data: pd.DataFrame, defects: set, **classifier_config: dict
) -> KerasClassifier:
    model = KerasClassifier(
        build_fn=build_mlp,
        input_dim=len(input_data.columns),
        optimizer=classifier_config["optimizer"],
        output_dim=len(defects),
        hidden_layer_sizes=classifier_config["hidden_layer_sizes"],
        epochs=classifier_config["epochs"],
        batch_size=classifier_config["batch_size"],
        verbose=3,
    )
    return model
