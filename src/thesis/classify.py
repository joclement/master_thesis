from pathlib import Path
import pickle
import sys
from typing import Final, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
import yaml

from . import __version__, data, models, util


class ClassificationHandler:
    def __init__(self, config):
        self.config = config

        self.output_dir = Path(self.config["general"]["output_dir"])
        self.output_dir.mkdir(exist_ok=False)
        if "version" in self.config and self.config["version"] != __version__:
            raise ValueError(
                "Non matching version {self.config['version']}, wanted {__version__}"
            )
        self.config["version"] = __version__
        self._save_config()

        self.calc_cm = self.config["general"]["calc_cm"]
        self.metric = self.config["general"]["metric"]
        self.cv = self.config["general"]["cv"]
        self.save_models = self.config["general"]["save_models"]

        iterables = [
            ["train", "val", "accuracy", "top_k_accuracy"],
            list(range(self.cv)),
        ]
        score_columns = pd.MultiIndex.from_product(iterables, names=["metric", "index"])
        self.scores = pd.DataFrame(
            index=self.config["models-to-run"], columns=score_columns
        )

        measurements, _ = data.read_recursive(self.config["general"]["data_dir"])
        self.y: Final = pd.Series(data.get_defects(measurements))
        data.clip_neg_pd_values(measurements)
        self.measurements: Final = [df.drop(data.CLASS, axis=1) for df in measurements]

        self.defect_names = [
            data.DEFECT_NAMES[data.Defect(d)] for d in sorted(set(self.y))
        ]

    def _save_config(self):
        with open(Path(self.output_dir, "config.yml"), "w") as outfile:
            yaml.dump(self.config, outfile)

    def _report_confusion_matrix(self, name: str, model_folder: Path, confusion_matrix):
        print(f"Confusion matrix for {name}:")
        print(
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

    def _get_measurements_copy(self):
        return [df.copy() for df in self.measurements]

    def _cross_validate(self, model_name, model_folder, classifier, X):
        skfold = StratifiedKFold(n_splits=self.cv)
        all_val_correct = []
        all_val_predictions = []
        for idx, split_indexes in enumerate(skfold.split(X, self.y)):
            train_index, val_index = split_indexes
            if isinstance(X, pd.DataFrame):
                X_train = X.loc[train_index, :]
                X_val = X.loc[val_index, :]
            else:
                X_train = X[train_index]
                X_val = X[val_index]
            y_train = self.y[train_index]
            y_val = self.y[val_index]

            classifier.fit(X_train, y_train)

            train_predictions = classifier.predict(X_train)
            val_proba_predictions = classifier.predict_proba(X_val)
            val_predictions = np.argmax(val_proba_predictions, axis=1)
            self.scores.loc[model_name, ("top_k_accuracy", idx)] = top_k_accuracy_score(
                y_val, val_proba_predictions, k=3
            )
            assert np.array_equal(val_predictions, classifier.predict(X_val))

            score = getattr(sklearn.metrics, self.metric)
            train_score = score(y_train, train_predictions)
            print("Train score: ", train_score)
            self.scores.loc[model_name, ("train", idx)] = train_score

            val_score = score(y_val, val_predictions)
            print("Validation score: ", val_score)
            self.scores.loc[model_name, ("val", idx)] = val_score
            self.scores.loc[model_name, ("accuracy", idx)] = accuracy_score(
                y_val, val_predictions
            )

            if self.calc_cm:
                confusion_matrix = metrics.confusion_matrix(y_val, val_predictions)
                self._report_confusion_matrix(str(idx), model_folder, confusion_matrix)
                all_val_predictions.extend(val_predictions)
                all_val_correct.extend(y_val)

        if self.calc_cm:
            confusion_matrix = metrics.confusion_matrix(
                all_val_correct, all_val_predictions
            )
            self._report_confusion_matrix(model_name, model_folder, confusion_matrix)

    def run(self):
        for model_name in self.config["models-to-run"]:
            print("Model: ", model_name)
            classifier, X = models.get_model_with_data(
                self._get_measurements_copy(),
                model_name,
                self.config["models"][model_name],
            )
            model_folder = Path(self.output_dir, model_name)
            self._cross_validate(model_name, model_folder, classifier, X)

            if self.save_models:
                classifier.fit(X, self.y)
                model_folder.mkdir(exist_ok=True)
                pickle.dump(classifier, open(Path(model_folder, "model.p"), "wb"))

            print("\n ============================================================ \n")
        self._finish()

    def _finish(self):
        print(self.scores)
        print(self.scores.mean(axis=1))
        self.scores.to_csv(Path(self.output_dir, "models_scores.csv"))
        self._plot_results()

    def _plot_results(self):
        title = (
            f"cv: {self.cv}, n: {len(self.measurements)}, n_defects: {len(set(self.y))}"
        )
        plt.figure(figsize=(20, 10))
        ax = self.scores.plot.bar(rot=30, title=title, ylabel=self.metric)
        ax.legend(loc=3)
        util.finish_plot("models_all_bar", self.output_dir, False)


def main(config_path: Union[str, Path] = sys.argv[1]):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    OUTPUT_DIRECTORY folder where plot(s) will be saved
    """
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    classificationHandler = ClassificationHandler(config)
    classificationHandler.run()
