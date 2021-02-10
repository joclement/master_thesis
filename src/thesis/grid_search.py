from typing import Dict
import warnings

import click
from sklearn.model_selection import GridSearchCV
import yaml

from . import __version__
from .classify import ClassificationHandler


class MyGridSearch(ClassificationHandler):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        for model_name in self.config["models-to-run"]:
            click.echo(f"Model: {model_name}")
            pipeline = self.modelHandler.get_model(model_name)

            grid_params: Dict[str, list] = {}
            grid_config = self.config["models"][model_name]["grid"]
            for step in grid_config:
                for param_key in grid_config[step]:
                    grid_params[f"{step}__{param_key}"] = grid_config[step][param_key]
            grid_search = GridSearchCV(
                pipeline,
                grid_params,
                scoring="balanced_accuracy",
                n_jobs=self.n_jobs,
                error_score="raise",
                verbose=1,
            )
            grid_search.fit(self.get_X(model_name), self.y)
            click.echo("Best params:")
            click.echo(grid_search.best_params_)


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
    grid_search = MyGridSearch(config)
    grid_search.run()
