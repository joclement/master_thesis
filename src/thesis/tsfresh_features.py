from pathlib import Path

import click
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from . import __version__, data, models


def calc_relevant_features(
    input_directory, n_jobs=1, output_file: Path = None
) -> pd.DataFrame:
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    """
    measurements, _ = data.read_recursive(input_directory)
    data.clip_neg_pd_values(measurements)
    y = pd.Series(data.get_defects(measurements))

    all_df = models.convert_to_tsfresh_dataset(measurements)

    extracted_features = extract_features(
        all_df,
        column_id="id",
        column_kind="kind",
        column_sort=data.TIME,
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        impute_function=impute,
        show_warnings=True,
        n_jobs=n_jobs,
    )
    if output_file:
        extracted_features.to_csv(output_file)
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
def main(input_directory, n_jobs=1, output_file=None):
    calc_relevant_features(input_directory, n_jobs, output_file)
