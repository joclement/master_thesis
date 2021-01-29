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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tsfresh.transformers import FeatureSelector
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC

from . import classifiers, data, prepared_data
from .classifiers import MyKerasClassifier
from .constants import TOP_K_ACCURACY


def _finger_tsfresh(
    measurements: List[pd.DataFrame], **config
) -> Tuple[pd.DataFrame, TransformerMixin]:
    extracted_features = pd.read_csv(
        config["tsfresh_data"], index_col=[data.PATH, prepared_data.PART]
    )
    indexes = [
        (df.attrs[data.PATH], df.attrs[prepared_data.PART]) for df in measurements
    ]
    X = extracted_features[extracted_features.index.isin(indexes)]
    X = X.reset_index(drop=True)
    tsfreshTransformer = FeatureSelector(
        fdr_level=config["fdr_level"],
        ml_task="classification",
        multiclass=True,
        n_jobs=config["n_jobs"],
    )
    return X, tsfreshTransformer


class SeqFingerScaler(TransformerMixin):
    def __init__(self, Scaler, **kwargs):
        self._scaler = Scaler(**kwargs)

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


def _get_transformer(
    classifier_id: str, data_id: str, **config
) -> Optional[TransformerMixin]:
    if data_id in ["oned", "twod"]:
        return None
    if "seqfinger_" in data_id:
        if config["normalize"]:
            if classifier_id == "knn_dtw":
                return SeqFingerScaler(MinMaxScaler)
            elif classifier_id == "svm_dtw":
                return SeqFingerScaler(StandardScaler)
            else:
                raise ValueError(f"classifier {classifier_id} not supported.")
        else:
            return None
    if "finger_" in data_id:
        if config["normalize"]:
            if classifier_id in ["mlp", "svm"]:
                return StandardScaler()
            else:
                return MinMaxScaler()
        else:
            return None
    raise ValueError(f"Data Representation '{data_id}' not supported.")


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

    def _get_tsfresh_data_and_transformer(
        self, data_id: str, **config
    ) -> Tuple[pd.DataFrame, TransformerMixin]:
        if data_id == "finger_tsfresh":
            data, transformer = _finger_tsfresh(self.measurements, **config)
        else:
            raise ValueError(f"model with {data_id} not supported.")
        return data, transformer

    def get_model_with_data(self, model_name: str) -> Tuple[Pipeline, pd.DataFrame]:
        model_config = self.models_config[model_name]

        parts = model_name.split("-")
        assert len(parts) == 2
        classifier_id = parts[0]
        data_id = parts[1]

        pipeline = []
        if "tsfresh" in data_id:
            input_data, transformer = self._get_tsfresh_data_and_transformer(
                data_id, **model_config
            )
            pipeline.append(("tsfresh_feature_selector", transformer))
        elif data_id in self.cache:
            input_data = self.cache[data_id]
        else:
            get_input_data = getattr(prepared_data, data_id)
            data_config = model_config["data"] if "data" in model_config else {}
            input_data = get_input_data(self._get_measurements_copy(), **data_config)
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
    top_3_accuracy = TopKCategoricalAccuracy(k=3, name=TOP_K_ACCURACY)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", top_3_accuracy],
    )
    return model


def get_mlp(defects: set, **classifier_config: dict) -> KerasClassifier:
    callbacks = []
    if classifier_config["stop_early"]:
        earlyStopping = EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=classifier_config["patience"],
            verbose=2,
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
        verbose=2,
        callbacks=callbacks,
    )
    return model
