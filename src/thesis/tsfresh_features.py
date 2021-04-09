from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import feature_extraction
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from . import __version__, data, prepared_data
from .constants import DEFAULT_DURATION, MIN_TIME_DIFF
from .data import TreatNegValues
from .prepared_data import MeasurementNormalizer, oned_func


def _convert_to_tsfresh_dataset(
    measurements: List[pd.DataFrame], duration, frequency
) -> pd.DataFrame:
    time_serieses = oned_func(
        measurements, **{"fix_duration": duration, "frequency": frequency}
    )
    time_serieses = np.reshape(time_serieses, (time_serieses.shape[0], -1))
    all_df = pd.DataFrame(
        data={
            "id": np.concatenate(
                [
                    np.full(time_serieses.shape[1], fill_value=i, dtype="int16")
                    for i in range(time_serieses.shape[0])
                ]
            ),
            "sort": np.tile(
                np.arange(time_serieses.shape[1], dtype="int32"),
                reps=time_serieses.shape[0],
            ),
            "value": np.ravel(time_serieses),
        }
    )
    all_df = all_df.loc[all_df["value"] > 0.0]
    assert all_df["id"].nunique() == len(measurements)
    return all_df


def save_extract_features(
    measurements: List[pd.DataFrame],
    n_jobs: int,
    output_file,
    splitted: bool,
    ParameterSet=feature_extraction.MinimalFCParameters,
    duration=DEFAULT_DURATION,
    frequency=MIN_TIME_DIFF,
):
    click.echo("INFO: _convert_to_tsfresh_dataset")
    all_df = _convert_to_tsfresh_dataset(measurements, duration, frequency)

    click.echo("INFO: extract_features")
    extracted_features = extract_features(
        all_df,
        column_id="id",
        column_kind=None,
        column_sort="sort",
        column_value=None,
        default_fc_parameters=ParameterSet(),
        impute_function=impute,
        show_warnings=False,
        n_jobs=n_jobs,
    )
    click.echo("INFO: save csv file")
    if output_file:
        paths = [Path(df.attrs[data.PATH]) for df in measurements]
        extracted_features[data.PATH] = paths
        index = [data.PATH]
        if splitted:
            parts = [df.attrs[prepared_data.PART] for df in measurements]
            extracted_features[prepared_data.PART] = parts
            index.append(prepared_data.PART)
        extracted_features.set_index(index, verify_integrity=True).to_csv(output_file)
    extracted_features.drop(
        columns=[data.PATH, prepared_data.PART], inplace=True, errors="ignore"
    )
    return extracted_features


def calc_relevant_features(
    extracted_features: pd.DataFrame,
    y: pd.Series,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    """
    click.echo(f"extracted_features.shape: {extracted_features.shape}")
    relevance_table = calculate_relevance_table(
        extracted_features,
        y,
        ml_task="classification",
        multiclass=True,
        n_significant=3,
        n_jobs=n_jobs,
    )
    relevance_table = relevance_table[relevance_table.relevant]
    p_value_columns = [c for c in relevance_table.columns if "p_value" in c]
    relevance_table["p_value"] = relevance_table.loc[:, p_value_columns].sum(axis=1)
    relevance_table.sort_values("p_value", inplace=True)
    click.echo("relevant_features:")
    relevant_columns = [c for c in relevance_table.columns if "relevant_" in c]
    click.echo(relevance_table[["p_value"]].to_string())
    return relevance_table[["p_value", *relevant_columns]]


@click.command()
@click.version_option(version=__version__)
@click.argument("input_directory", type=click.Path(exists=True))
@click.option(
    "--n_jobs",
    "-j",
    default=1,
    show_default=True,
    help="Number of jobs",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(exists=False),
    help="Save extracted features",
)
@click.option(
    "--parameter_set",
    "-p",
    default="MinimalFCParameters",
    show_default=True,
    help="Choose tsfresh parameter set",
)
def main(
    input_directory,
    n_jobs,
    output_file,
    parameter_set,
):
    measurements, _ = data.read_recursive(input_directory, TreatNegValues.zero)
    click.echo("INFO: normalize measurements")
    measurements = MeasurementNormalizer().transform(measurements)
    click.echo("INFO: split measurements")
    measurements = prepared_data.adapt_durations(
        measurements, max_duration=DEFAULT_DURATION, step_duration="30 seconds"
    )

    y = pd.Series(data.get_defects(measurements))

    ParameterSet = getattr(feature_extraction, parameter_set)
    click.echo("INFO: save_extract_features")
    extracted_features = save_extract_features(
        measurements,
        n_jobs,
        output_file,
        ParameterSet,
    )
    calc_relevant_features(extracted_features, y, n_jobs)
