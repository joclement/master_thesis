from typing import Final, List, Tuple

import imblearn
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import numpy as np
from pyts.classification import BOSSVS
from pyts.classification import KNeighborsClassifier as PytsKNN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tsfresh.transformers import FeatureSelector
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC

from . import prepared_data
from .classifiers import (
    FastDtwKnn,
    LukasMeanDist,
    MyKerasClassifier,
    PolarityKNN,
    UnderSampleKNN,
    UndersampleTimeSeriesSVM,
)
from .constants import RANDOM_STATE
from .prepared_data import DimAdder, NormalizationMethod, Reshaper


def get_classifier(pipeline: Pipeline) -> BaseEstimator:
    return pipeline.named_steps["classifier"]


def no_sample_weight(classifier: BaseEstimator) -> bool:
    return isinstance(
        classifier,
        (BOSSVS, PytsKNN, KNeighborsClassifier, KNeighborsTimeSeriesClassifier),
    )


def no_predict_proba(classifier: BaseEstimator) -> bool:
    return isinstance(classifier, (SVC, TimeSeriesSVC, BOSSVS, RidgeClassifierCV))


def is_data_finger(data_id: str):
    return "finger_" in data_id and "seqfinger" not in data_id


def is_model_finger(model_name: str):
    return is_data_finger(split_model_name(model_name)[1])


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
        return {"Scaler": str(self.Scaler)}

    def set_params(self, **parameters):
        if "Scaler" in parameters:
            self.Scaler = parameters["Scaler"]
            self._scaler = self.Scaler()
        return self


def get_oned_scaler(normalizationMethod: NormalizationMethod):
    if normalizationMethod is NormalizationMethod.minmax:
        return TimeSeriesScalerMinMax()
    if normalizationMethod is NormalizationMethod.zscore:
        return TimeSeriesScalerMeanVariance()


def get_finger_scaler(normalizationMethod: NormalizationMethod):
    if normalizationMethod is NormalizationMethod.minmax:
        return MinMaxScaler()
    if normalizationMethod is NormalizationMethod.zscore:
        return StandardScaler()


def get_seqfinger_scaler(normalizationMethod: NormalizationMethod):
    if normalizationMethod is NormalizationMethod.minmax:
        return SeqFingerScaler(MinMaxScaler)
    if normalizationMethod is NormalizationMethod.zscore:
        return SeqFingerScaler(StandardScaler)


def add_scaler(
    pipeline: List[Tuple[str, TransformerMixin]],
    classifier_id: str,
    data_id: str,
    normalizationMethod: NormalizationMethod,
) -> None:
    if normalizationMethod is NormalizationMethod.none:
        return
    if "seqfinger_" in data_id:
        scaler = get_seqfinger_scaler(normalizationMethod)
    elif "finger_" in data_id or "tsfresh" == data_id:
        scaler = get_finger_scaler(normalizationMethod)
    elif data_id == "oned":
        scaler = get_oned_scaler(normalizationMethod)
    pipeline.append(("scaler", scaler))


def split_model_name(model_name: str):
    parts = model_name.split("-")
    if len(parts) != 2:
        raise ValueError("Invalid input: {model_name}.")
    classifier_id = parts[0]
    data_id = parts[1]
    return classifier_id, data_id


def add_selector(
    pipeline: List[Tuple[str, TransformerMixin]], model_config: dict
) -> None:
    if "select" in model_config:
        select_config = model_config["select"]
        if "variance" in select_config and select_config["variance"]:
            pipeline.append(("variance_selector", VarianceThreshold()))
        if "rfecv" in select_config:
            svc = SVC(kernel="linear")
            rfecv = RFECV(
                estimator=svc,
                scoring="balanced_accuracy",
                min_features_to_select=select_config["rfecv"]["min_features"],
                cv=select_config["rfecv"]["cv"],
            )
            pipeline.append(("selector", rfecv))
        if "rfe" in select_config and "rfecv" not in select_config:
            svc = SVC(kernel="linear")
            rfe = RFE(
                estimator=svc,
                n_features_to_select=select_config["rfe"]["features"],
            )
            pipeline.append(("selector", rfe))
        if "tsfresh" in select_config:
            tsfresh_selector = FeatureSelector(**select_config["tsfresh"])
            pipeline.append(("selector", tsfresh_selector))
        if "kbest" in select_config:
            kbest_selector = SelectKBest(**select_config["kbest"])
            pipeline.append(("selector", kbest_selector))
        if "frommodel" in select_config:
            frommodel = SelectFromModel(
                LinearSVC(dual=False, **select_config["frommodel"])
            )
            pipeline.append(("selector", frommodel))


class ModelHandler:
    def __init__(
        self,
        defects: set,
        models_config: dict,
        verbose: bool,
    ):
        self.defects: Final = defects
        self.models_config: Final = models_config
        self.verbose = verbose

    def get_model(self, model_name: str) -> Pipeline:
        model_config = self.models_config[model_name]

        classifier_id, data_id = split_model_name(model_name)

        pipeline = []

        data_config = model_config["data"] if "data" in model_config else {}
        get_feature_builder = getattr(prepared_data, data_id)
        data_transformer = get_feature_builder(**data_config)
        if isinstance(data_transformer, list):
            for t in data_transformer:
                pipeline.append(t)
        else:
            pipeline.append((data_id, data_transformer))

        add_selector(pipeline, model_config)
        add_scaler(
            pipeline,
            classifier_id,
            data_id,
            NormalizationMethod(model_config["normalize"]),
        )

        if "reshaper" in model_config and model_config["reshaper"] is True:
            pipeline.append(("reshaper", Reshaper()))

        if "undersample" in model_config:
            undersample_config = model_config["undersample"]
            if undersample_config is True:
                undersampler = RandomUnderSampler(random_state=RANDOM_STATE)
            elif type(undersample_config) is dict:
                undersampler = RandomUnderSampler(
                    sampling_strategy=undersample_config, random_state=RANDOM_STATE
                )
            pipeline.append(("undersample", undersampler))

        if (
            "reshaper" in model_config
            and model_config["reshaper"] is True
            and "classifier_id" in ["knn_dtw"]
        ):
            pipeline.append(("dim_adder", DimAdder()))

        add_classifier(
            pipeline,
            classifier_id,
            self.defects,
            model_config["classifier"] if "classifier" in model_config else {},
        )

        if "undersample" in model_config and model_config["undersample"] is not False:
            return imblearn.pipeline.Pipeline(pipeline, verbose=self.verbose)
        return Pipeline(pipeline, verbose=self.verbose)


CLASSIFIER_MAP = {
    "bossvs": BOSSVS,
    "dt": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "knn": KNeighborsClassifier,
    "knn_dtw": KNeighborsTimeSeriesClassifier,
    "knn_fastdtw": FastDtwKnn,
    "pytsknn": PytsKNN,
    "lr": LogisticRegression,
    "ott_algo": LukasMeanDist,
    "svm": SVC,
    "svm_dtw": TimeSeriesSVC,
    "lgbm": LGBMClassifier,
    "uknn_dtw": UnderSampleKNN,
    "usvm_dtw": UndersampleTimeSeriesSVM,
    "polknn_dtw": PolarityKNN,
    "ridgecv": RidgeClassifierCV,
}


def add_classifier(
    pipeline: List[Tuple[str, TransformerMixin]],
    classifier_id: str,
    defects: set,
    classifier_config: dict,
) -> BaseEstimator:
    if classifier_id == "mlp":
        classifier = get_mlp(defects, **classifier_config)
    else:
        if classifier_id not in {
            "bossvs",
            "pytsknn",
            "ott_algo",
            "knn",
            "knn_dtw",
            "knn_fastdtw",
            "uknn_dtw",
            "usvm_dtw",
            "polknn_dtw",
            "ridgecv",
        }:
            classifier_config["random_state"] = RANDOM_STATE
        classifier = CLASSIFIER_MAP[classifier_id](**classifier_config)
    pipeline.append(("classifier", classifier))


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layer_sizes: Tuple[int],
    optimizer: str,
    dropout: float,
) -> KerasClassifier:
    model = Sequential()
    model.add(keras.Input(shape=(input_dim,)))
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation="relu"))
        model.add(Dropout(dropout))
    model.add(
        Dense(output_dim, activation="softmax"),
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
    )
    return model


def get_mlp(defects: set, **classifier_config: dict) -> KerasClassifier:
    model = MyKerasClassifier(
        build_fn=build_mlp,
        optimizer=classifier_config["optimizer"],
        output_dim=len(defects),
        hidden_layer_sizes=classifier_config["hidden_layer_sizes"],
        dropout=classifier_config["dropout"],
        epochs=classifier_config["epochs"],
        batch_size=classifier_config["batch_size"],
        verbose=classifier_config["verbose"],
    )
    return model
