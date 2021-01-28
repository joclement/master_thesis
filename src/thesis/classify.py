from pathlib import Path
import pickle
import shutil
from typing import Final, List, Optional
import warnings

import click
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tslearn.svm import TimeSeriesSVC
import yaml

from . import __version__, data, models, util
from .constants import (
    ACCURACY_SCORE,
    CONFIG_FILENAME,
    CONFIG_MODELS_RUN_ID,
    METRIC_NAMES,
    SCORES_FILENAME,
    TOP_K_ACCURACY,
    TRAIN_ACCURACY,
    TRAIN_SCORE,
    VAL_SCORE,
)
from .prepared_data import split_by_durations
from .util import to_dataTIME
from .visualize_results import plot_results


def _print_score(name: str, value: float) -> None:
    click.echo(f"{name}: {value:.2f}")


def get_classifier(pipeline: Pipeline) -> BaseEstimator:
    return list(pipeline.named_steps.values())[-1]


def is_keras(pipeline: Pipeline) -> bool:
    return isinstance(get_classifier(pipeline), (KerasClassifier))


def adapt_durations(
    measurements: List[pd.DataFrame],
    min_duration: str,
    max_duration: str,
    split: bool,
    drop_empty: bool,
):
    min_duration = pd.Timedelta(min_duration)
    long_enough_measurements = []
    for df in measurements:
        if df[data.TIME_DIFF].sum() > to_dataTIME(min_duration):
            long_enough_measurements.append(df)

    if not split:
        return long_enough_measurements
    return split_by_durations(
        long_enough_measurements, pd.Timedelta(max_duration), drop_empty
    )


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
        self._save_config()

        self.calc_cm = self.config["general"]["calc_cm"]
        self.metric = getattr(sklearn.metrics, self.config["general"]["metric"])

        self.cv = self.config["general"]["cv"]
        self.save_models = self.config["general"]["save_models"]

        all_score_names = {TRAIN_SCORE, VAL_SCORE, *METRIC_NAMES}
        iterables = [all_score_names, list(range(self.cv))]
        score_columns = pd.MultiIndex.from_product(iterables, names=["metric", "index"])
        self.scores = pd.DataFrame(
            index=self.config[CONFIG_MODELS_RUN_ID], columns=score_columns, dtype=float
        )

        measurements, _ = data.read_recursive(self.config["general"]["data_dir"])
        if len(measurements) == 0:
            raise ValueError(f"No data in: {self.config['general']['data_dir']}")
        measurements = self._keep_wanted_defects(measurements)
        data.clip_neg_pd_values(measurements)
        measurements = adapt_durations(
            measurements,
            config["general"]["min_duration"],
            config["general"]["max_duration"],
            config["general"]["split"],
            config["general"]["drop_empty"],
        )
        self.y: Final = pd.Series(data.get_defects(measurements))
        self.onehot_y = LabelBinarizer().fit_transform(self.y)
        self.modelHandler = models.ModelHandler(
            measurements,
            self.y,
            self.config["models"],
            self.config["general"]["write_cache"],
            self.get_cache_path(),
        )

        self.defect_names = [
            data.DEFECT_NAMES[data.Defect(d)] for d in sorted(set(self.y))
        ]
        skfold = StratifiedKFold(n_splits=self.cv)
        self.cv_splits = list(skfold.split(np.zeros(len(self.y)), self.y))

        self.finished = False

    def __del__(self):
        if hasattr(self, "finished") and not self.finished:
            self.scores.to_csv(Path(self.output_dir, SCORES_FILENAME))

    def get_cache_path(self) -> Optional[Path]:
        return (
            Path(self.config["general"]["cache_path"])
            if "cache_path" in self.config["general"]
            else None
        )

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

    def _do_val_predictions(self, model_name, idx, pipeline, X_val, y_val):
        if isinstance(get_classifier(pipeline), (SVC, TimeSeriesSVC)):
            val_predictions = pipeline.predict(X_val)
        else:
            val_proba_predictions = pipeline.predict_proba(X_val)
            val_predictions = np.argmax(val_proba_predictions, axis=1)
            top_k_accuracy = top_k_accuracy_score(y_val, val_proba_predictions, k=3)
            _print_score(TOP_K_ACCURACY, top_k_accuracy)
            self.scores.loc[model_name, (TOP_K_ACCURACY, idx)] = top_k_accuracy
        return val_predictions

    def _cross_validate(self, model_name, model_folder, pipeline, X):
        all_val_correct = []
        all_val_predictions = []
        for idx, split_indexes in enumerate(self.cv_splits):
            click.echo()
            click.echo(f"cv: {idx}")
            train_index, val_index = split_indexes
            if isinstance(X, pd.DataFrame):
                X_train = X.loc[train_index, :]
                X_val = X.loc[val_index, :]
            else:
                X_train = X[train_index]
                X_val = X[val_index]
            y_train = self.y[train_index]
            y_val = self.y[val_index]
            if is_keras(pipeline):
                class_weights = dict(
                    enumerate(
                        compute_class_weight("balanced", np.unique(y_train), y_train)
                    )
                )
                pipeline.fit(
                    X_train,
                    self.onehot_y[train_index],
                    classifier__validation_data=(X_val, self.onehot_y[val_index]),
                    classifier__class_weight=class_weights,
                )
                if self.config["general"]["show_plots"]:
                    sns.lineplot(data=get_classifier(pipeline).history.history)
                    plt.show()
            else:
                pipeline.fit(X_train, y_train)

            train_predictions = pipeline.predict(X_train)
            val_predictions = self._do_val_predictions(
                model_name, idx, pipeline, X_val, y_val
            )
            assert np.array_equal(val_predictions, pipeline.predict(X_val))

            train_score = self.metric(y_train, train_predictions)
            _print_score("Train score", train_score)
            self.scores.loc[model_name, (TRAIN_SCORE, idx)] = train_score
            train_accuracy = accuracy_score(y_train, train_predictions)
            _print_score("Train accuracy score", train_accuracy)
            self.scores.loc[model_name, (TRAIN_ACCURACY, idx)] = train_accuracy

            val_score = self.metric(y_val, val_predictions)
            _print_score("Val score", val_score)
            self.scores.loc[model_name, (VAL_SCORE, idx)] = val_score
            accuracy = accuracy_score(y_val, val_predictions)
            _print_score("Val accuracy score", accuracy)
            self.scores.loc[model_name, (ACCURACY_SCORE, idx)] = accuracy

            if self.calc_cm:
                click.echo()
                confusion_matrix = metrics.confusion_matrix(y_val, val_predictions)
                self._report_confusion_matrix(str(idx), model_folder, confusion_matrix)
                all_val_predictions.extend(val_predictions)
                all_val_correct.extend(y_val)

        click.echo()
        _print_score(
            "Val score", self.scores.loc[model_name, (VAL_SCORE, slice(None))].mean()
        )
        _print_score(
            "Val accuracy",
            self.scores.loc[model_name, (ACCURACY_SCORE, slice(None))].mean(),
        )
        _print_score(
            "Val top 3 accuracy",
            self.scores.loc[model_name, (TOP_K_ACCURACY, slice(None))].mean(),
        )
        if self.calc_cm:
            click.echo()
            confusion_matrix = metrics.confusion_matrix(
                all_val_correct, all_val_predictions
            )
            self._report_confusion_matrix(model_name, model_folder, confusion_matrix)

    def run(self):
        for model_name in self.config["models-to-run"]:
            click.echo(f"Model: {model_name}")
            pipeline, X = self.modelHandler.get_model_with_data(model_name)
            model_folder = Path(self.output_dir, model_name)
            self._cross_validate(model_name, model_folder, pipeline, X)

            if self.save_models:
                pipeline.fit(X, self.y)
                model_folder.mkdir(exist_ok=True)
                if is_keras(pipeline):
                    get_classifier(pipeline).model.save(Path(model_folder, "model.h5"))
                else:
                    pickle.dump(pipeline, open(Path(model_folder, "model.p"), "wb"))

            click.echo(
                "\n ============================================================ \n"
            )
        self._finish()

    def _finish(self):
        click.echo(self.scores)
        click.echo(self.scores.loc[:, (VAL_SCORE, slice(None))].mean(axis=1))
        self.scores.to_csv(Path(self.output_dir, SCORES_FILENAME))
        self.finished = True
        description = f"cv: {self.cv}, n: {len(self.y)}, n_defects: {len(set(self.y))}"
        plot_results(self.scores, self.output_dir, description=description)


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
        config = yaml.safe_load(stream)

    if not warn:
        warnings.simplefilter("ignore")
    classificationHandler = ClassificationHandler(config)
    classificationHandler.run()
