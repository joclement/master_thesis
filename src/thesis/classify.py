from pathlib import Path
import sys
from typing import Final, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, cross_val_score
import yaml

from . import __version__, data, models, util


class ClassificationHandler:
    def __init__(self, config):
        self.config = config

        self.output_dir = Path(self.config["general"]["output_dir"])
        if "version" in self.config and self.config["version"] != __version__:
            raise ValueError(
                "Non matching version {self.config['version']}, wanted {__version__}"
            )
        self.config["version"] = __version__
        self._save_config()

        self.calc_cm = self.config["general"]["calc_cm"]
        self.metric = self.config["general"]["metric"]
        self.cv = self.config["general"]["cv"]
        self.n_jobs = self.config["general"]["n_jobs"]

        score_columns = [f"cv{idx}" for idx in range(self.config["general"]["cv"])]
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

    def _report_confusion_matrix(self, model_name: str, confusion_matrix):
        print(f"Confusion matrix for {model_name}:")
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
        filestem = f"confusion_matrix_{model_name}"
        filestem = filestem.replace(" ", "_")
        util.finish_plot(filestem, self.output_dir)

    def _get_measurements_copy(self):
        return [df.copy() for df in self.measurements]

    def _cross_validate(self, model_name, model_config):
        pipeline, X = models.get_model_with_data(
            self._get_measurements_copy(), model_name, model_config
        )
        self.scores.loc[model_name, :] = cross_val_score(
            pipeline,
            X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            error_score="raise",
            n_jobs=self.n_jobs,
        )
        print(f"Scores for {model_name}: {self.scores.loc[model_name, :]}")

        if self.calc_cm:
            confusion_matrix = metrics.confusion_matrix(
                self.y,
                cross_val_predict(pipeline, X, self.y, cv=self.cv, n_jobs=self.n_jobs),
            )
            self._report_confusion_matrix(model_name, confusion_matrix)

    def run(self):
        for model_name in self.config["models-to-run"]:
            model_config = self.config["models"][model_name]
            self._cross_validate(model_name, model_config)
            print("\n ============================================================ \n")
        self._finish()

    def _finish(self):
        print(self.scores)
        print(self.scores.mean(axis=1))
        self.scores.to_csv(Path(self.output_dir, f"models_{self.metric}.csv"))
        self._plot_results()

    def _plot_results(self):
        title = (
            f"metric: {self.metric}, cv: {self.cv}, n: {len(self.measurements)}"
            f" , n_defects: {len(set(self.y))}"
        )
        plt.figure(figsize=(20, 10))
        ax = self.scores.plot.bar(rot=30, title=title, ylabel=self.metric)
        ax.legend(loc=3)
        util.finish_plot(f"models_{self.metric}_bar", self.output_dir, False)


def main(config_path: Union[str, Path] = sys.argv[1]):
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    OUTPUT_DIRECTORY folder where plot(s) will be saved
    """
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)

    classificationHandler = ClassificationHandler(config)
    classificationHandler.run()
