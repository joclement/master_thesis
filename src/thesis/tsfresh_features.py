from pathlib import Path
from typing import List

import click
import pandas as pd
from tsfresh import extract_features
from tsfresh import feature_extraction
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from . import __version__, data, prepared_data
from .prepared_data import split_by_durations


def _convert_to_tsfresh_dataset(measurements: List[pd.DataFrame]) -> pd.DataFrame:
    measurements = [m.loc[:, [data.TIME_DIFF, data.PD]] for m in measurements]
    for index, df in enumerate(measurements):
        df["id"] = index
        df["kind"] = data.PD
        df["sort"] = df[data.TIME_DIFF].cumsum()
    all_df = pd.concat(measurements)
    assert all_df["id"].nunique() == len(measurements)
    all_df = all_df.rename(columns={data.PD: "value"})
    return all_df


def save_extract_features(
    measurements: List[pd.DataFrame],
    n_jobs: int,
    output_file,
    splitted: bool,
    ParameterSet=feature_extraction.MinimalFCParameters,
):
    all_df = _convert_to_tsfresh_dataset(measurements)

    extracted_features = extract_features(
        all_df,
        column_id="id",
        column_kind="kind",
        column_sort="sort",
        column_value="value",
        default_fc_parameters=ParameterSet(),
        impute_function=impute,
        show_warnings=True,
        n_jobs=n_jobs,
    )
    if output_file:
        paths = [Path(df.attrs[data.PATH]) for df in measurements]
        extracted_features[data.PATH] = paths
        index = [data.PATH]
        if splitted:
            parts = [df.attrs[prepared_data.PART] for df in measurements]
            extracted_features[prepared_data.PART] = parts
            index.append(prepared_data.PART)
        extracted_features.set_index(index, verify_integrity=True).to_csv(output_file)
    extracted_features.drop(columns=data.PATH, inplace=True, errors="ignore")
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
        n_jobs=n_jobs,
    )
    relevance_table = relevance_table[relevance_table.relevant]
    p_value_columns = [c for c in relevance_table.columns if "p_value" in c]
    relevance_table["p_value"] = relevance_table.loc[:, p_value_columns].sum(axis=1)
    relevance_table.sort_values("p_value", inplace=True)
    click.echo("relevant_features:")
    relevant_columns = [c for c in relevance_table.columns if "relevant_" in c]
    click.echo(relevance_table[["p_value", *relevant_columns]])
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
@click.option(
    "--duration",
    "-d",
    type=str,
    help="Set duration to split measurement files by",
)
@click.option("--drop/--no-drop", default=False, help="Drop empty frames")
def main(
    input_directory,
    n_jobs=1,
    output_file=None,
    parameter_set="MinimalFCParameters",
    duration="",
    drop: bool = False,
):
    measurements, _ = data.read_recursive(input_directory)

    data.clip_neg_pd_values(measurements)
    if duration:
        measurements = split_by_durations(
            measurements, pd.Timedelta(duration), drop_empty=drop
        )

    y = pd.Series(data.get_defects(measurements))

    ParameterSet = getattr(feature_extraction, parameter_set)
    extracted_features = save_extract_features(
        measurements,
        n_jobs,
        output_file,
        True if duration else False,
        ParameterSet,
    )
    calc_relevant_features(extracted_features, y, n_jobs)
