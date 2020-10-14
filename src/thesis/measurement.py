from enum import Enum
from pathlib import Path

import pandas as pd

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


class Defect(Enum):
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


def read(filepath):
    experiment = pd.read_csv(filepath, sep=SEPERATOR, decimal=DECIMAL_SIGN)
    experiment[TIMEDIFF] = experiment[TIME].diff()

    filename = Path(filepath).stem
    experiment = _add_voltage_sign(experiment, filename)
    experiment[CLASS] = _get_defect(filename)

    return experiment


def read_recursive(dir_path) -> list:
    measurements = []
    for csv_file in Path(dir_path).rglob("*.csv"):
        measurements.append(read(csv_file))

    return measurements
