from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TIMEDIFF = "TimeDiff [s]"
PD = "A [nV]"
TEST_VOLTAGE = "Voltage [kV]"
TIME = "Time [s]"

SEPERATOR = ";"
DECIMAL_SIGN = ","

VOLTAGE_SIGN = "Voltage Sign"
POS_VOLTAGE = "(+DC)"
NEG_VOLTAGE = "(-DC)"

CLASS = "Defect"


class VoltageSign(Enum):
    positive = 1
    negative = 0


class Defect(IntEnum):
    free_particle = 0
    particle_insulator = 1
    protrusion = 2
    free_potential = 3


def _add_voltage_sign(df, filename: str) -> VoltageSign:
    if filename[: len(POS_VOLTAGE)] == POS_VOLTAGE:
        df[VOLTAGE_SIGN] = VoltageSign.positive
    elif filename[: len(NEG_VOLTAGE)] == NEG_VOLTAGE:
        df[VOLTAGE_SIGN] = VoltageSign.negative
    return df


def _get_defect(filename: str) -> Defect:
    if "Partikel" in filename:
        if "Isolator" in filename:
            return Defect.particle_insulator
        else:
            return Defect.free_particle
    elif "Spitze an Erde" in filename or "Spitze an HS" in filename:
        return Defect.protrusion
    elif "freies Potential" in filename:
        return Defect.free_potential
    else:
        raise ValueError(f"No knwown defect found: {filename}")


def _do_sanity_test(df: pd.DataFrame):
    if df[TIME].min() < 0.0 or not df[TIME].equals(df[TIME].sort_values()):
        raise ValueError("Time values seem corrupt.")


def read(filepath) -> pd.DataFrame:
    experiment = pd.read_csv(filepath, sep=SEPERATOR, decimal=DECIMAL_SIGN)
    experiment[TIMEDIFF] = experiment[TIME].diff()

    filename = Path(filepath).stem
    experiment = _add_voltage_sign(experiment, filename)
    experiment[CLASS] = _get_defect(filename)

    _do_sanity_test(experiment)

    return experiment


def read_recursive(dir_path) -> Tuple[List[pd.DataFrame], list]:
    measurements = []
    csv_filepaths = list(Path(dir_path).rglob("*.csv"))
    for csv_filepath in csv_filepaths:
        measurements.append(read(csv_filepath))

    return measurements, csv_filepaths


class Normalizer:
    def __init__(self, measurements: List[pd.DataFrame]):
        self.max_pd = max([measurement[PD].max() for measurement in measurements])
        self.max_timediff = max(
            [measurement[TIMEDIFF].max() for measurement in measurements]
        )

    def apply(self, measurements: List[pd.DataFrame]):
        for measurement in measurements:
            measurement[PD] = measurement[PD] / self.max_pd
        measurement[TIMEDIFF] = measurement[TIMEDIFF] / self.max_timediff


def split_train_test(
    measurements: List[pd.DataFrame], test_size=0.2
) -> Tuple[list, list]:
    classes = [df[CLASS][0] for df in measurements]
    return train_test_split(measurements, test_size=test_size, stratify=classes)
