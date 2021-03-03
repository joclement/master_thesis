from pathlib import Path
import pickle
from typing import Dict
import warnings

import click
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
import yaml

from . import __version__
from .classify import ClassificationHandler
from .metrics import file_scores


FINGERPRINT_COMPARE_GRID = [
    {
        "scaler": [StandardScaler()],
        "classifier__batch_size": [1, 10],
        "classifier__dropout": [0.0, 0.05, 0.2],
        "classifier__hidden_layer_sizes": [(5,), (20,), (5, 3), (20, 10)],
        "classifier__epochs": [50, 100, 150],
    },
    {
        "scaler": [MinMaxScaler()],
        "classifier": [KNeighborsClassifier()],
        "classifier__weights": ["uniform", "distance"],
        "classifier__n_neighbors": [1, 5, 10, 30],
    },
    {
        "scaler": ["passthrough"],
        "classifier": [RandomForestClassifier()],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__bootstrap": [True, False],
        "classifier__class_weight": ["balanced"],
    },
    {
        "scaler": [StandardScaler(), MinMaxScaler()],
        "classifier": [SVC()],
        "classifier__decision_function_shape": ["ovr", "ovo"],
        "classifier__class_weight": ["balanced"],
    },
    {
        "scaler": ["passthrough"],
        "classifier": [LGBMClassifier()],
        "classifier__feature_fraction": [1.0, 0.8],
        "classifier__min_split_gain": [0.0, 0.01, 0.1],
        "classifier__max_depth": [10, 50],
        "classifier__num_leaves": [10, 30],
    },
]


class MyGridSearch(ClassificationHandler):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        for model_name in self.config["models-to-run"]:
            click.echo(f"Model: {model_name}")
            pipeline = self.modelHandler.get_model(model_name)

            grid_params: Dict[str, list] = {}
            grid_config = self.config["models"][model_name]["grid"]
            if grid_config == "fingerprint_compare":
                grid_params = FINGERPRINT_COMPARE_GRID
            else:
                for step in grid_config:
                    for param_key in grid_config[step]:
                        grid_params[f"{step}__{param_key}"] = grid_config[step][
                            param_key
                        ]
            grid_search = GridSearchCV(
                pipeline,
                grid_params,
                cv=self.cv_splits,
                scoring=make_scorer(file_scores),
                n_jobs=self.n_jobs,
                error_score="raise",
                verbose=1,
            )
            grid_search.fit(self.get_X(model_name), self.y)
            click.echo("Best params:")
            click.echo(grid_search.best_params_)
            model_folder = Path(self.output_dir, model_name)
            model_folder.mkdir(exist_ok=True)
            with open(Path(model_folder, "grid-search-results.p"), "wb") as file:
                pickle.dump(grid_search.cv_results_, file)


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
    grid_search = MyGridSearch(config)
    grid_search.run()
