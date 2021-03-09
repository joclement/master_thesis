from datetime import datetime
import os
from pathlib import Path
import pickle
import random
import shutil
from typing import Any, Dict, Final, Iterable, List, Optional, Tuple, Union
import warnings

import click
from keras.wrappers.scikit_learn import KerasClassifier
import lightgbm
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from pyts.classification import BOSSVS
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.utils import estimator_html_repr
from sklearn.utils.class_weight import compute_class_weight
import tensorflow
from tslearn.svm import TimeSeriesSVC
import yaml

from . import __version__, data, util
from .constants import (
    ACCURACY_SCORE,
    AVG_FILE_SCORE,
    BALANCED_ACCURACY_SCORE,
    CONFIG_FILENAME,
    CONFIG_MODELS_RUN_ID,
    DataPart,
    FILE_SCORE,
    MODEL_ID,
    PART,
    PREDICTIONS_FILENAME,
    SCORES_FILENAME,
    TOP_K_ACCURACY_SCORE,
)
from .data import TreatNegValues
from .fingerprint import (
    CATEGORICAL_FEATURES,
    get_categorical_features,
    get_feature_names,
    get_X_index,
)
from .metrics import avg_file_scores, file_scores, top_k_accuracy_score
from .models import is_data_finger, is_model_finger, ModelHandler
from .prepared_data import adapt_durations, extract_features, MeasurementNormalizer

SEED: Final = 23
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)


def combine(dataPart: DataPart, metric_name: str):
    return f"{dataPart}_{metric_name}"


def get_classifier(pipeline: Pipeline) -> BaseEstimator:
    return list(pipeline.named_steps.values())[-1]


def get_data_transformer(pipeline: Pipeline) -> BaseEstimator:
    return list(pipeline.named_steps.values())[0]


def get_categorical_features_info(
    transformer: Union[FeatureUnion, FunctionTransformer], X: pd.DataFrame
) -> Tuple[Iterable[str], Iterable[str]]:
    if isinstance(transformer, FeatureUnion):
        return get_feature_names(transformer), get_categorical_features(transformer)
    return list(X.columns), CATEGORICAL_FEATURES


def is_keras(pipeline: Pipeline) -> bool:
    return isinstance(get_classifier(pipeline), KerasClassifier)


def group_by_file(measurements: List[pd.DataFrame]) -> List[int]:
    filenames = [df.attrs[data.PATH] for df in measurements]
    groups = []
    index = 0
    current_filename = filenames[0]
    seen = []
    for filename in filenames:
        assert filename not in seen
        if filename != current_filename:
            seen.append(current_filename)
            index += 1
            current_filename = filename
        groups.append(index)

    return groups


def build_index(measurements: List[pd.DataFrame]) -> pd.Index:
    index = [get_X_index(df) for df in measurements]
    if isinstance(index[0], tuple):
        return pd.MultiIndex.from_tuples(index, name=[data.PATH, PART])
    else:
        return pd.Index(data=index, name=data.PATH)


def get_index(X) -> pd.Index:
    if isinstance(X, pd.DataFrame):
        return X.index
    elif isinstance(X, list):
        return build_index(X)


def get_X_part(X, index):
    if isinstance(X, pd.DataFrame):
        return X.iloc[index]
    elif isinstance(X, list):
        return [X[idx] for idx in index]
    else:
        raise ValueError("Invalid X.")


class ClassificationHandler:
    def __init__(self, config):
        pd.set_option("precision", 2)
        self.config = config

        self.output_dir = Path(self.config["general"]["output_dir"])
        if self.config["general"]["overwrite_output_dir"] and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=False)
        if "version" in self.config and self.config["version"] != __version__:
            raise ValueError(
                "Non matching version {self.config['version']}, wanted {__version__}"
            )
        self.config["version"] = __version__
        self.config["datetime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_config()

        self.n_jobs = self.config["general"]["n_jobs"]
        self.calc_cm = self.config["general"]["calc_cm"]

        durationAdapter = FunctionTransformer(adapt_durations)
        preprocessor = [("adapt_durations", durationAdapter)]
        preprocessor = Pipeline(preprocessor)
        self.measurements: Final = preprocessor.set_params(
            adapt_durations__kw_args={
                "min_len": self.config["general"]["min_len"],
                "max_len": self.config["general"]["max_len"],
                "repeat": self.config["general"]["repeat"],
                "min_duration": self.config["general"]["min_duration"],
                "step_duration": self.config["general"]["step_duration"],
                "max_duration": self.config["general"]["max_duration"],
                "split": self.config["general"]["split"],
                "drop_empty": self.config["general"]["drop_empty"],
            },
        ).fit_transform(self.get_measurements())
        if self.config["general"]["save_models"]:
            with open(Path(self.output_dir, "preprocessor.p"), "wb") as file:
                pickle.dump(preprocessor, file)

        self.y: Final = pd.Series(
            data=data.get_defects(self.measurements),
            index=build_index(self.measurements),
            dtype=np.int8,
        )
        self.defects: Final = sorted(set(self.y))
        self.defect_names: Final = [
            data.DEFECT_NAMES[data.Defect(d)] for d in self.defects
        ]
        self.cv_splits: Final = self._generate_cv_splits()

        if any([is_model_finger(m) for m in self.config["models-to-run"]]):
            finger_preprocessor = [
                ("extract_features", FunctionTransformer(extract_features)),
            ]
            if self.config["general"]["normalize_fingerprints"]:
                finger_preprocessor.insert(0, ("normalize", MeasurementNormalizer()))
            finger_preprocessor = Pipeline(finger_preprocessor)
            click.echo("Calc finger features...")
            self.finger_X = finger_preprocessor.fit_transform(self.measurements)
            click.echo("Done.")
            if self.config["general"]["save_models"]:
                with open(Path(self.output_dir, "finger_preprocessor.p"), "wb") as file:
                    pickle.dump(finger_preprocessor, file)
        self.modelHandler = ModelHandler(
            self.defects, self.config["models"], self.config["general"]["verbose"]
        )

        metric_names = sorted(
            [
                BALANCED_ACCURACY_SCORE,
                ACCURACY_SCORE,
                TOP_K_ACCURACY_SCORE,
            ]
        )
        dataParts = [DataPart.val]
        if self.config["general"]["calc_train_score"]:
            dataParts.append(DataPart.train)
        all_score_names = sorted(
            [combine(p, m) for p in dataParts for m in metric_names]
        )
        if self.config["general"]["cv"] in ["group", "logo"]:
            all_score_names.append(combine(DataPart.val, AVG_FILE_SCORE))
            all_score_names.append(combine(DataPart.val, FILE_SCORE))
        iterables = [all_score_names, list(range(len(self.cv_splits)))]
        score_columns = pd.MultiIndex.from_product(iterables, names=["metric", "index"])
        self.scores = pd.DataFrame(
            index=self.config[CONFIG_MODELS_RUN_ID],
            columns=score_columns,
            dtype=np.float16,
        )

        self.predictions = pd.DataFrame(
            data=-1,
            columns=self.config[CONFIG_MODELS_RUN_ID],
            index=build_index(self.measurements),
            dtype=np.int8,
        )

    def get_measurements(self) -> List[pd.DataFrame]:
        measurements, _ = data.read_recursive(
            self.config["general"]["data_dir"],
            TreatNegValues(self.config["general"]["treat_negative_values"]),
        )
        return self._keep_wanted_defects(measurements)

    def group_by_file_and_class(self) -> List[int]:
        pairs = [
            (df.attrs[data.PATH], df.attrs[data.CLASS]) for df in self.measurements
        ]
        pairs = list(dict.fromkeys(pairs))
        file_defects = pd.Series([pair[1] for pair in pairs])
        file_defect_counts = file_defects.value_counts(ascending=True)
        num_of_groups = file_defect_counts.iloc[0]

        current_filename = ""
        seen = []
        groups = []
        counts_per_defect = {defect: 0 for defect in file_defect_counts.index}
        for filename, defect in zip(
            [df.attrs[data.PATH] for df in self.measurements], self.y
        ):
            assert filename not in seen
            if filename != current_filename:
                seen.append(current_filename)
                index = counts_per_defect[defect] % num_of_groups
                counts_per_defect[defect] += 1
                current_filename = filename
            groups.append(index)

        return groups

    def _generate_cv_splits(self):
        cv = self.config["general"]["cv"]
        if isinstance(cv, int):
            cross_validator = StratifiedKFold(n_splits=cv, shuffle=True)
            groups = None
        elif cv == "group":
            cross_validator = LeaveOneGroupOut()
            groups = self.group_by_file_and_class()
        elif cv == "logo":
            cross_validator = LeaveOneGroupOut()
            groups = group_by_file(self.measurements)
        else:
            raise ValueError("Invalid cv.")

        return list(cross_validator.split(np.zeros(len(self.y)), self.y, groups))

    def _keep_wanted_defects(self, measurements: List[pd.DataFrame]):
        defects = [data.Defect[defect] for defect in self.config["defects"]]
        measurements = [df for df in measurements if df.attrs[data.CLASS] in defects]
        if len(measurements) == 0:
            raise ValueError(f"No data matching these defects: {defects}")
        return measurements

    def _save_config(self):
        with open(Path(self.output_dir, CONFIG_FILENAME), "w") as outfile:
            yaml.dump(self.config, outfile)

    def _report_confusion_matrix(self, name: str, model_folder: Path, confusion_matrix):
        click.echo(f"Confusion matrix for {name}:")
        click.echo(
            pd.DataFrame(
                confusion_matrix,
                index=self.defect_names,
                columns=self.defect_names,
            ).to_string()
        )

        metrics.ConfusionMatrixDisplay(
            confusion_matrix, display_labels=self.defect_names
        ).plot()
        model_folder.mkdir(exist_ok=True)
        util.finish_plot(f"confusion_matrix_{name}", model_folder)

    def _train(
        self,
        pipeline: Pipeline,
        X: Union[pd.DataFrame, List[pd.DataFrame]],
        train_index: range,
        val_index: range,
    ):
        X_train = get_X_part(X, train_index)
        y_train = self.y[train_index]
        if is_keras(pipeline):
            class_weights = dict(
                enumerate(compute_class_weight("balanced", np.unique(y_train), y_train))
            )
            pipeline.fit(
                X_train,
                y_train,
                classifier__class_weight=class_weights,
            )
        elif isinstance(get_classifier(pipeline), LGBMClassifier):
            X_val = get_X_part(X, val_index)
            y_val = self.y[val_index]
            fit_params: Dict[str, Any] = {}
            if self.config["general"]["show_plots"]:
                fit_params["classifier__eval_set"] = [
                    (get_data_transformer(pipeline).transform(X_val), y_val),
                    (get_data_transformer(pipeline).transform(X_train), y_train),
                ]
            if is_data_finger(list(pipeline.named_steps.keys())[0]):
                feature_name, categorical_feature = get_categorical_features_info(
                    get_data_transformer(pipeline), X
                )
                fit_params.update(
                    {
                        "classifier__feature_name": feature_name,
                        "classifier__categorical_feature": categorical_feature,
                    }
                )
            pipeline.fit(
                X_train,
                y_train,
                **fit_params,
            )
        else:
            pipeline.fit(X_train, y_train)
        if (
            isinstance(get_classifier(pipeline), LGBMClassifier)
            and self.config["general"]["show_plots"]
        ):
            lightgbm.plot_metric(get_classifier(pipeline))
            util.finish_plot(None, None, True)

    def _calc_scores(self, y_true, predictions) -> pd.Series:
        return pd.Series(
            data={
                BALANCED_ACCURACY_SCORE: balanced_accuracy_score(y_true, predictions),
                ACCURACY_SCORE: accuracy_score(y_true, predictions),
            },
        )

    def calc_scores(
        self,
        pipeline: Pipeline,
        X,
        y_true: pd.Series,
        dataPart: DataPart,
        model_name: Optional[str] = None,
    ) -> Tuple[pd.Series, np.ndarray]:
        if isinstance(get_classifier(pipeline), (SVC, TimeSeriesSVC, BOSSVS)):
            predictions = pipeline.predict(X)
            scores = self._calc_scores(y_true, predictions)
        else:
            proba_predictions = pipeline.predict_proba(X)
            predictions = np.argmax(proba_predictions, axis=1)
            scores = self._calc_scores(y_true, predictions)
            scores[TOP_K_ACCURACY_SCORE] = top_k_accuracy_score(
                y_true, proba_predictions, self.defects
            )
            if (
                self.config["general"]["cv"] in ["logo", "group"]
                and dataPart is DataPart.val
            ):
                scores[AVG_FILE_SCORE] = avg_file_scores(y_true, proba_predictions)
        if dataPart is DataPart.val:
            self.predictions.loc[get_index(X), model_name] = predictions
        if (
            self.config["general"]["cv"] in ["logo", "group"]
            and dataPart is DataPart.val
        ):
            scores[FILE_SCORE] = file_scores(y_true, predictions)
        if dataPart is DataPart.val and self.calc_cm:
            self._all_val_predictions.extend(predictions)
            self._all_val_correct.extend(y_true)
        return scores.sort_index(), metrics.confusion_matrix(y_true, predictions)

    def assign_and_print_scores(
        self, model_name: str, dataPart: DataPart, idx: int, scores: pd.Series
    ):
        scores = scores.rename(lambda score_name: combine(dataPart, score_name))
        click.echo()
        click.echo(scores)
        self.scores.loc[
            model_name, pd.IndexSlice[scores.index.tolist(), idx]
        ] = scores.values

    def _cross_validate(
        self, model_name: str, model_folder: Path, pipeline: Pipeline, X
    ):
        self._all_val_correct: List[data.Defect] = []
        self._all_val_predictions: List[data.Defect] = []
        for idx, split_indexes in enumerate(self.cv_splits):
            click.echo()
            click.echo(f"cv: {idx}")
            train_index, val_index = split_indexes

            click.echo("train...")
            self._train(pipeline, X, train_index, val_index)
            click.echo("Done.")

            if self.config["general"]["calc_train_score"]:
                train_scores, _ = self.calc_scores(
                    pipeline,
                    get_X_part(X, train_index),
                    self.y[train_index],
                    DataPart.train,
                )
                self.assign_and_print_scores(
                    model_name, DataPart.train, idx, train_scores
                )

            val_scores, confusion_matrix = self.calc_scores(
                pipeline,
                get_X_part(X, val_index),
                self.y[val_index],
                DataPart.val,
                model_name,
            )
            self.assign_and_print_scores(model_name, DataPart.val, idx, val_scores)
            if self.calc_cm and self.config["general"]["cv"] != "logo":
                click.echo()
                self._report_confusion_matrix(str(idx), model_folder, confusion_matrix)

        click.echo("\nMean scores:")
        click.echo(self.scores.loc[model_name].groupby(level=0).mean())

        if self.calc_cm:
            click.echo()
            confusion_matrix = metrics.confusion_matrix(
                self._all_val_correct, self._all_val_predictions
            )
            self._report_confusion_matrix(model_name, model_folder, confusion_matrix)

    def run(self):
        for model_name in self.config["models-to-run"]:
            click.echo(f"Model: {model_name}")
            pipeline = self.modelHandler.get_model(model_name)
            model_folder = Path(self.output_dir, model_name)

            self._cross_validate(
                model_name, model_folder, pipeline, self.get_X(model_name)
            )

            model_folder.mkdir(exist_ok=True)
            with open(Path(model_folder, "pipeline_representation.html"), "w") as file:
                file.write(estimator_html_repr(pipeline))
            self.scores.to_csv(Path(self.output_dir, SCORES_FILENAME))
            self.predictions.to_csv(Path(self.output_dir, PREDICTIONS_FILENAME))
            if self.config["general"]["save_models"]:
                self._save_models(pipeline, model_folder, model_name)
            click.echo(
                "\n ============================================================ \n"
            )

    def get_X(self, model_name: str):
        if is_model_finger(model_name):
            return self.finger_X
        else:
            return self.measurements

    def _save_models(
        self, pipeline: Pipeline, model_folder: Path, model_name: str
    ) -> None:
        pipeline.fit(self.get_X(model_name), self.y)
        if is_keras(pipeline):
            pipeline_steps = list(
                zip(pipeline.named_steps.keys(), pipeline.named_steps.values())
            )
            for index, named_step in enumerate(pipeline_steps[:-1]):
                name, transformer = named_step
                with open(
                    Path(model_folder, f"pipeline_step{index}_{name}.p"), "wb"
                ) as file:
                    pickle.dump(transformer, file)
            get_classifier(pipeline).model.save(Path(model_folder, "model.h5"))
        else:
            with open(Path(model_folder, f"{MODEL_ID}-{model_name}.p"), "wb") as file:
                pickle.dump(pipeline, file)


@click.command()
@click.version_option(version=__version__)
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--warn/--no-warn", default=False)
def main(config_path, warn):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    OUTPUT_DIRECTORY folder where plot(s) will be saved
    """
    with open(config_path, "r") as stream:
        config = yaml.load(stream)

    if not warn:
        warnings.simplefilter("ignore")
    classificationHandler = ClassificationHandler(config)
    classificationHandler.run()
