from pathlib import Path
import sys
from typing import List, Union

import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute

from . import data


def convert_to_tsfresh_dataset(measurements: List[pd.DataFrame]) -> pd.DataFrame:
    measurements = [m.loc[:, [data.TIME, data.PD]] for m in measurements]
    measurements = [_convert_to_time_series(df, i) for i, df in enumerate(measurements)]
    min_len = min([len(m) for m in measurements])
    measurements = [df[: min(len(df.index), 10 * min_len)] for df in measurements]
    return pd.concat(measurements)


def _convert_to_time_series(df: pd.DataFrame, index: int) -> pd.DataFrame:
    df.loc[:, data.TIME] = pd.to_datetime(df[data.TIME], unit=data.TIME_UNIT)
    df.set_index(data.TIME, inplace=True)
    time_series = df.rename(columns={data.PD: "value"})
    time_series["kind"] = data.PD
    time_series["id"] = index
    return time_series


def main(input_directory: Union[str, Path] = sys.argv[1]) -> pd.DataFrame:
    """Print measurement info on given measurement file or folder

    INPUT_DIRECTORY folder containing csv files for classification
    """
    measurements, _ = data.read_recursive(input_directory)
    data.clip_neg_pd_values(measurements)
    y = pd.Series(data.get_defects(measurements))

    all_df = convert_to_tsfresh_dataset(measurements)

    extracted_features = extract_features(
        all_df,
        column_id="id",
        column_kind="kind",
        column_value="value",
        impute_function=impute,
        show_warnings=True,
    )
    print("extracted_features.shape: ", extracted_features.shape)
    relevance_table = calculate_relevance_table(
        extracted_features,
        y,
        ml_task="classification",
        multiclass=True,
        show_warnings=True,
    )
    relevance_table = relevance_table[relevance_table.relevant]
    p_value_columns = [c for c in relevance_table.columns if "p_value" in c]
    relevance_table["p_value"] = relevance_table.loc[:, p_value_columns].sum(axis=1)
    relevance_table.sort_values("p_value", inplace=True)
    print("relevant_features:")
    relevant_columns = [c for c in relevance_table.columns if "relevant_" in c]
    print(relevance_table[["p_value", *relevant_columns]])
    return relevance_table[["p_value", *relevant_columns]]
