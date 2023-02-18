from typing import Callable, List

import click
import pandas as pd

from . import __version__
from .fingerprint import (
    get_feature_names,
    ott_feature_union,
    own_feature_union,
    relown_feature_union,
    tugraz_feature_union,
)


FEATURE_SETS: List[Callable] = [
    ott_feature_union,
    own_feature_union,
    relown_feature_union,
    tugraz_feature_union,
]


def get_weibull_combined(feature_name: str):
    if feature_name[-1] == "\u03B1":
        weibull_b = list(feature_name)
        weibull_b[-1] = "\u03B2"
        return feature_name + "+" + "".join(weibull_b)
    elif feature_name[-1] == "\u03B2":
        weibull_a = list(feature_name)
        weibull_a[-1] = "\u03B1"
        return "".join(weibull_a) + "+" + feature_name


def adapt_weibull_feature_names(feature_names: List[str]) -> List[str]:
    feature_names = [
        get_weibull_combined(name) if "Weib" in name else name for name in feature_names
    ]
    return list(set(feature_names))


def main(prediction_times: pd.DataFrame) -> None:
    for feature_set_function in FEATURE_SETS:
        feature_names = adapt_weibull_feature_names(
            get_feature_names(feature_set_function())
        )
        sum_per_feature = prediction_times[feature_names].sum(axis=0)
        click.echo(f"{feature_set_function.__name__}: {sum_per_feature.sum()}")


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "prediction_times_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
def click_command(prediction_times_file):
    main(pd.read_csv(prediction_times_file, header=0, index_col="path"))
