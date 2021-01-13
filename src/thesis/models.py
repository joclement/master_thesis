from typing import Dict, Final, List, Optional, Tuple, Union

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import tsfresh.feature_extraction
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.transformers import RelevantFeatureAugmenter
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.svm import TimeSeriesSVC

from . import classifiers, data, prepared_data


def convert_to_tsfresh_dataset(measurements: List[pd.DataFrame]) -> pd.DataFrame:
    measurements = [m.loc[:, [data.TIME, data.PD]] for m in measurements]
    for index, df in enumerate(measurements):
        df["id"] = index
        df["kind"] = data.PD
    all_df = pd.concat(measurements)
    all_df = all_df.rename(columns={data.PD: "value"})
    return all_df


def _seqfinger_tsfresh(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    # TODO do own implementation
    return None, None


def _finger_tsfresh(measurements: List[pd.DataFrame], **config) -> pd.DataFrame:
    X_data = convert_to_tsfresh_dataset(measurements)
    if "default_fc_parameters" in config:
        DefaultFcParameters = getattr(
            tsfresh.feature_extraction, config["default_fc_parameters"]
        )
    else:
        DefaultFcParameters = ComprehensiveFCParameters
    tsfreshTransformer = RelevantFeatureAugmenter(
        column_id="id",
        column_kind="kind",
        column_sort=data.TIME,
        column_value="value",
        fdr_level=config["fdr_level"],
        ml_task="classification",
        multiclass=True,
        n_jobs=config["n_jobs"],
        default_fc_parameters=DefaultFcParameters(),
    )
    tsfreshTransformer.set_timeseries_container(X_data)
    X = pd.DataFrame(index=list(range(len(measurements))))
    return X, tsfreshTransformer


def _get_seqfinger_transformer(normalize: bool) -> Optional[TransformerMixin]:
    if normalize:
        return SeqFingerMinMaxScaler()
    else:
        return None


class SeqFingerMinMaxScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(**kwargs)

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


def _get_transformer(data_id: str, **config) -> Optional[TransformerMixin]:
    if data_id in ["oned", "twod"]:
        return None
    if "seqfinger_" in data_id:
        return _get_seqfinger_transformer(config["normalize"])
    if "finger_" in data_id:
        return MinMaxScaler()
    raise ValueError(f"Data Representation '{data_id}' not supported.")


class ModelHandler:
    def __init__(
        self,
        measurements: List[pd.DataFrame],
        y: Union[pd.Series, np.array],
        models_config: dict,
    ):
        self.measurements: Final = measurements
        self.y: Final = y
        self.models_config: Final = models_config
        self.cache: Dict[str, pd.DataFrame] = {}

    def _get_measurements_copy(self):
        return [df.copy() for df in self.measurements]

    def _get_tsfresh_data_and_transformer(
        self, data_id: str, **config
    ) -> Tuple[pd.DataFrame, TransformerMixin]:
        if data_id == "finger_tsfresh":
            data, transformer = _finger_tsfresh(self.measurements, **config)
        elif data_id == "seqfinger_tsfresh":
            data, transformer = _seqfinger_tsfresh(self.measurements, **config)
        return data, transformer

    def get_model_with_data(self, model_name: str) -> Tuple[Pipeline, pd.DataFrame]:
        model_config = self.models_config[model_name]

        parts = model_name.split("-")
        assert len(parts) == 2
        classifier_id = parts[0]
        data_id = parts[1]

        if "tsfresh" in data_id:
            input_data, transformer = self._get_tsfresh_data_and_transformer(
                data_id, **model_config
            )
        elif data_id in self.cache:
            input_data = self.cache[data_id]
            transformer = _get_transformer(data_id, **model_config)
        else:
            get_input_data = getattr(prepared_data, data_id)
            input_data = get_input_data(self._get_measurements_copy(), **model_config)
            transformer = _get_transformer(data_id, **model_config)
            self.cache[data_id] = input_data

        if "classifier_hyperparameters" in model_config:
            classifier_config = model_config["classifier_hyperparameters"]
        else:
            classifier_config = {}
        classifier = get_classifier(
            classifier_id,
            input_data,
            set(self.y),
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
    **classifier_config: dict,
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
