from pathlib import Path
import pickle
from typing import Dict, Final, List, Optional, Tuple, Union

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.metrics import TopKCategoricalAccuracy
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC

from . import classifiers, data, prepared_data
from .classifiers import MyKerasClassifier
from .constants import K, TOP_K_ACCURACY_SCORE


class SeqFingerScaler(TransformerMixin, BaseEstimator):
    def __init__(self, Scaler, **kwargs):
        self.Scaler = Scaler
        self._scaler = self.Scaler()

    def fit(self, X, y=None, **kwargs):
        X = np.array(X)
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, y=None, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

    def get_params(self, deep=True):
        return {"Scaler": self.Scaler}

    def set_params(self, **parameters):
        if "Scaler" in parameters:
            self.Scaler = parameters["Scaler"]
            self._scaler = self.Scaler()
        return self


def _get_transformer(
    classifier_id: str, data_id: str, **config
) -> Optional[TransformerMixin]:
    if data_id == "twod" or not config["normalize"]:
        return None
    if "seqfinger_" in data_id:
        if classifier_id == "knn_dtw":
            return SeqFingerScaler(MinMaxScaler)
        elif classifier_id == "svm_dtw":
            return SeqFingerScaler(StandardScaler)
        else:
            raise ValueError(f"classifier {classifier_id} not supported.")
    if "finger_" in data_id:
        if classifier_id in ["mlp", "svm"]:
            return StandardScaler()
        elif classifier_id in ["knn", "dt"]:
            return MinMaxScaler()
        else:
            raise ValueError(f"classifier {classifier_id} not supported.")
    if data_id == "oned":
        return TimeSeriesScalerMinMax()
    raise ValueError(f"Data Representation '{data_id}' not supported.")


def split_model_name(model_name: str):
    parts = model_name.split("-")
    assert len(parts) == 2
    classifier_id = parts[0]
    data_id = parts[1]
    return classifier_id, data_id


class ModelHandler:
    def __init__(
        self,
        measurements: List[pd.DataFrame],
        y: Union[pd.Series, np.array],
        models_config: dict,
        write_cache: bool,
        cache_path: Optional[Path],
    ):
        self.measurements: Final = measurements
        self.y: Final = y
        self.models_config: Final = models_config
        self.write_cache = write_cache
        self.cache: Dict[str, pd.DataFrame] = {}
        if cache_path:
            self.cache_path = cache_path
            if self.cache_path.exists():
                with open(self.cache_path, "rb") as cache_file:
                    self.cache = pickle.load(cache_file)

    def __del__(self):
        if self.write_cache and hasattr(self, "cache_path"):
            with open(self.cache_path, "wb") as cache_file:
                pickle.dump(self.cache, cache_file)

    def _get_measurements_copy(self):
        return [df.copy() for df in self.measurements]

    def get_model_with_data(self, model_name: str) -> Tuple[Pipeline, pd.DataFrame]:
        model_config = self.models_config[model_name]

        classifier_id, data_id = split_model_name(model_name)

        pipeline = []
        if data_id in self.cache:
            input_data = self.cache[data_id]
        else:
            get_input_data = getattr(prepared_data, data_id)
            data_config = model_config["data"] if "data" in model_config else {}
            feature_generator = FunctionTransformer(get_input_data)
            feature_generator.set_params(kw_args=data_config)
            input_data = feature_generator.fit_transform(self._get_measurements_copy())
            self.cache[data_id] = input_data
        scaler = _get_transformer(classifier_id, data_id, **model_config)
        if scaler:
            pipeline.append(("scaler", scaler))

        if "classifier" in model_config:
            classifier_config = model_config["classifier"]
        else:
            classifier_config = {}
        classifier = get_classifier(
            classifier_id,
            set(self.y),
            **classifier_config,
        )
        pipeline.append(("classifier", classifier))

        return Pipeline(pipeline), input_data


def get_classifier(
    classifier_id: str,
    defects: set,
    **classifier_config: dict,
) -> BaseEstimator:
    if classifier_id == "dt":
        return DecisionTreeClassifier(**classifier_config)
    if classifier_id == "knn":
        return KNeighborsClassifier(**classifier_config)
    if classifier_id == "knn_dtw":
        return KNeighborsTimeSeriesClassifier(**classifier_config)
    if classifier_id == "mlp":
        return get_mlp(defects, **classifier_config)
    if classifier_id == "ott_algo":
        return classifiers.LukasMeanDist()
    if classifier_id == "svm":
        return SVC(**classifier_config)
    if classifier_id == "svm_dtw":
        return TimeSeriesSVC()


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layer_sizes: List[int],
    optimizer: str,
    dropout: float,
) -> KerasClassifier:
    model = Sequential()
    neurons_per_layer = hidden_layer_sizes
    model.add(keras.Input(shape=(input_dim,)))
    model.add(Dense(neurons_per_layer[0], activation="relu"))
    model.add(Dropout(dropout))
    model.add(
        Dense(output_dim, activation="softmax"),
    )
    top_k_accuracy = TopKCategoricalAccuracy(k=K, name=TOP_K_ACCURACY_SCORE)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", top_k_accuracy],
    )
    return model


def get_mlp(defects: set, **classifier_config: dict) -> KerasClassifier:
    callbacks = []
    if classifier_config["stop_early"]:
        earlyStopping = EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=classifier_config["patience"],
            verbose=classifier_config["verbose"],
            mode="auto",
            restore_best_weights=True,
        )
        callbacks.append([earlyStopping])
    model = MyKerasClassifier(
        build_fn=build_mlp,
        optimizer=classifier_config["optimizer"],
        output_dim=len(defects),
        hidden_layer_sizes=classifier_config["hidden_layer_sizes"],
        dropout=classifier_config["dropout"],
        epochs=classifier_config["epochs"],
        batch_size=classifier_config["batch_size"],
        verbose=classifier_config["verbose"],
        callbacks=callbacks,
    )
    return model
