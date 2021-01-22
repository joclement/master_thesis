from pathlib import Path
from typing import Final, List

import click
import pandas as pd
from tsfresh import extract_features
from tsfresh import feature_extraction
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from . import __version__, data, prepared_data
from .prepared_data import split_by_durations

INDEX: Final = [data.PATH, prepared_data.PART]
CHUNK_SIZE: Final = 4


def _convert_to_tsfresh_dataset(measurements: List[pd.DataFrame]) -> List[pd.DataFrame]:
    measurements = [m.loc[:, [data.TIME_DIFF, data.PD]] for m in measurements]
    dfs = []
    for index, df in enumerate(measurements):
        df["id"] = index
        df["kind"] = data.PD
        df["sort"] = df[data.TIME_DIFF].cumsum()
        df.rename(columns={data.PD: "value"}, inplace=True)
    for i in range(0, len(measurements), CHUNK_SIZE):
        dfs.append(pd.concat(measurements[i : i + CHUNK_SIZE]))
    assert sum([df["id"].nunique() for df in dfs]) == len(measurements)
    return dfs


def _load(output_file: Path) -> pd.DataFrame:
    return pd.read_csv(output_file, index_col=INDEX)


def _save(
    paths: List[Path],
    parts: List[int],
    extracted_features: pd.DataFrame,
    output_file: Path,
) -> None:
    extracted_features[data.PATH] = paths
    extracted_features[prepared_data.PART] = parts
    extracted_features.set_index(INDEX, verify_integrity=True, inplace=True)
    if output_file.exists():
        existing_extracted_features = _load(output_file)
        pd.concat([existing_extracted_features, extracted_features]).to_csv(output_file)
    else:
        extracted_features.to_csv(output_file)


def save_extract_features(
    measurements: List[pd.DataFrame],
    n_jobs: int,
    output_file: Path,
    ParameterSet=feature_extraction.MinimalFCParameters,
):
    dfs = _convert_to_tsfresh_dataset(measurements)

    paths = [Path(df.attrs[data.PATH]) for df in measurements]
    if all([prepared_data.PART in df.attrs for df in measurements]):
        parts = [df.attrs[prepared_data.PART] for df in measurements]
    elif not any([prepared_data.PART in df.attrs for df in measurements]):
        parts = [0 for df in measurements]
    else:
        raise ValueError("Invalid measurements.")

    index = 0
    for df in dfs:
        click.echo("Process:")
        for path, part in zip(
            paths[index : index + CHUNK_SIZE], parts[index : index + CHUNK_SIZE]
        ):
            click.echo(path)
            click.echo(part)
        click.echo(f"len df: {len(df.index)}")
        extracted_features = extract_features(
            df,
            column_id="id",
            column_kind="kind",
            column_sort="sort",
            column_value="value",
            default_fc_parameters=ParameterSet(),
            impute_function=impute,
            show_warnings=False,
            chunksize=1,
            n_jobs=n_jobs,
        )
        assert len(extracted_features.index) == CHUNK_SIZE or df.equals(dfs[-1])
        if output_file:
            _save(
                paths[index : index + CHUNK_SIZE],
                parts[index : index + CHUNK_SIZE],
                extracted_features,
                output_file,
            )
        index += CHUNK_SIZE
    extracted_features = _load(output_file)
    extracted_features.drop(
        columns=[data.PATH, prepared_data.PART], inplace=True, errors="ignore"
    )
    return extracted_features.reset_index(drop=True)


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
    required=True,
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
    output_file = Path(output_file)
    if output_file.exists():
        raise ValueError("output_file exists.")
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
        ParameterSet,
    )
    calc_relevant_features(extracted_features, y, n_jobs)
