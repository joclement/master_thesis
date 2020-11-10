from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Tuple

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


class Defect(IntEnum):
    free_particle = 0
    particle_insulator = 1
    protrusion_earth = 2
    protrusion_hv = 3
    floating = 4


# TODO try this maybe as an improvement:
# https://stackoverflow.com/questions/43862184/associating-string-representations-with-an-enum-that-uses-integer-values
DEFECT_NAMES = {
    Defect.free_particle: "Free Particle",
    Defect.particle_insulator: "Particle on Insulator",
    Defect.protrusion_earth: "Protrusion on Earth",
    Defect.protrusion_hv: "Protrusion on HV",
    Defect.floating: "Floating",
}


def _add_voltage_sign(df, filename: str) -> VoltageSign:
    if filename[: len(POS_VOLTAGE)] == POS_VOLTAGE:
        df[VOLTAGE_SIGN] = VoltageSign.positive
    elif filename[: len(NEG_VOLTAGE)] == NEG_VOLTAGE:
        df[VOLTAGE_SIGN] = VoltageSign.negative
    return df


def _get_defect(filename: str) -> Defect:
    if "Spitze an Erde" in filename:
        return Defect.protrusion_earth
    elif "Spitze an HS" in filename:
        return Defect.protrusion_hv
    elif "freies Potential" in filename:
        return Defect.floating
    elif "Isolator" in filename:
        return Defect.particle_insulator
    elif "Partikel" in filename:
        return Defect.free_particle
    else:
        raise ValueError(f"No knwown defect found: {filename}")


def _do_sanity_test(df: pd.DataFrame, filepath):
    if TIME not in df or PD not in df:
        raise ValueError(f"TIME or PD column missing in file: {filepath}")

    if (TEST_VOLTAGE in df and len(df.columns) > 6) or len(df.columns) > 5:
        raise ValueError(f"Unexpected columns in file: {filepath}")

    if df[TIME].min() < 0.0 or not df[TIME].equals(df[TIME].sort_values()):
        raise ValueError(f"Time values are corrupt in file: {filepath}")


def read(filepath) -> pd.DataFrame:
    experiment = pd.read_csv(filepath, sep=SEPERATOR, decimal=DECIMAL_SIGN)
    _do_sanity_test(experiment, filepath)

    experiment[TIMEDIFF] = experiment[TIME].diff()

    filename = Path(filepath).stem
    experiment = _add_voltage_sign(experiment, filename)
    experiment[CLASS] = _get_defect(filename)

    # @note: The PD values have been probably converted in in a different unit,
    #        namely V and not and not nV.
    experiment[PD] *= 10 ** 9

    return experiment


def clip_neg_pd_values(measurements: List[pd.DataFrame]):
    for measurement in measurements:
        measurement[PD].clip(lower=0, inplace=True)


def read_recursive(dir_path) -> Tuple[List[pd.DataFrame], list]:
    measurements = []
    csv_filepaths = list(Path(dir_path).rglob("*.csv"))
    for csv_filepath in csv_filepaths:
        measurements.append(read(csv_filepath))

    return measurements, csv_filepaths


def get_defects(measurements: List[pd.DataFrame]):
    return [measurement[CLASS][0] for measurement in measurements]
