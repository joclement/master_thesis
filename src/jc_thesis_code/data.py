from enum import Enum, IntEnum
from pathlib import Path
import textwrap
from typing import Dict, Final, List, Tuple, Union

import pandas as pd


PD: Final = "A [nV]"

TEST_VOLTAGE: Final = "Voltage [kV]"

TIME_IN_FILE: Final = "Time [s]"
TIME_UNIT: Final = "ms"
TIME_DIFF: Final = f"TimeDiff [{TIME_UNIT}]"
START_TIME: Final = f"Start time [{TIME_UNIT}]"

SEPERATOR: Final = ";"
DECIMAL_SIGN: Final = ","

VOLTAGE_SIGN: Final = "Voltage Sign"
POS_VOLTAGE: Final = "+DC"
NEG_VOLTAGE: Final = "-DC"

CLASS: Final = "Defect"
PATH: Final = "path"


class VoltageSign(IntEnum):
    positive = 1
    negative = 0

    def __str__(self):
        return _VOLTAGE_NAMES[self.value]

    @staticmethod
    def from_str(voltage_sign: str):
        return {value: key for key, value in _VOLTAGE_NAMES.items()}[voltage_sign]


_VOLTAGE_NAMES: Final[Dict[IntEnum, str]] = {
    VoltageSign.positive: POS_VOLTAGE,
    VoltageSign.negative: NEG_VOLTAGE,
}


class Defect(IntEnum):
    free_particle = 0
    particle_insulator = 1
    protrusion_earth = 2
    protrusion_hv = 3
    floating = 4
    cavity = 5
    noise = 6

    def __str__(self):
        return _DEFECT_NAMES[self.value]

    def wrapped(self):
        return textwrap.fill(str(self), 12)

    def abbreviation(self):
        return textwrap.fill(_DEFECT_ABBREVIATIONS[self.value], 10)


_DEFECT_NAMES: Final = {
    Defect.free_particle: "Moving Particle",
    Defect.particle_insulator: "Particle on Insulator",
    Defect.protrusion_earth: "Protrusion on Enclosure",
    Defect.protrusion_hv: "Protrusion on HV",
    Defect.floating: "Floating Electrode",
    Defect.cavity: "Void in Insulator",
    Defect.noise: "Noise",
}

_DEFECT_ABBREVIATIONS: Final = {
    Defect.free_particle: "Moving Particle",
    Defect.particle_insulator: "Particle Insulator",
    Defect.protrusion_earth: "Protrusion Enclosure",
    Defect.protrusion_hv: "Protrusion HV",
    Defect.floating: "Floating",
    Defect.cavity: "Void",
    Defect.noise: "Noise",
}


class TreatNegValues(Enum):
    nothing = "nothing"
    zero = "zero"
    absolute = "absolute"


def get_names(defects: Union[List[Defect], pd.Series]) -> List[str]:
    return [str(Defect(defect)) for defect in defects]


def get_abbreviations(defects: Union[List[Defect], pd.Series]) -> List[str]:
    return [Defect(defect).abbreviation() for defect in defects]


def _has_voltage_sign(voltage_sign: VoltageSign, filename: str) -> bool:
    volt_name = str(voltage_sign)
    return (
        filename[1:].startswith(volt_name)
        or filename[11:].startswith(volt_name)
        or filename[16:].startswith(volt_name)
        or filename[17:].startswith(volt_name)
    )


def _get_voltage_sign(filename: str) -> VoltageSign:
    detected_voltages_signs = []
    if _has_voltage_sign(VoltageSign.positive, filename):
        detected_voltages_signs.append(VoltageSign.positive)
    if _has_voltage_sign(VoltageSign.negative, filename):
        detected_voltages_signs.append(VoltageSign.negative)
    if len(detected_voltages_signs) == 0:
        raise ValueError(f"No voltage sign found: {filename}")
    if len(detected_voltages_signs) > 1:
        raise ValueError(f"Both voltage signs found: {filename}")
    return detected_voltages_signs[0]


def get_defect(filename: str) -> Defect:
    defects = []
    if (
        "Störeinkopplungen" in filename
        or "Stützer_Störer" in filename
        or "nach Gasfuellen" in filename
    ):
        defects.append(Defect.noise)
    elif "Stütze" in filename:
        defects.append(Defect.cavity)
    if "Spitze an Erde" in filename or "Spitze_an_Erde" in filename:
        defects.append(Defect.protrusion_earth)
    if (
        "Spitze an HS" in filename
        or "Spitze-HSP0" in filename
        or "-Spitze-HS_" in filename
    ):
        defects.append(Defect.protrusion_hv)
    if (
        "freies Potential" in filename
        or "freies_Potential" in filename
        or "Floating_Hülse" in filename
        or "Floating electrode" in filename
    ):
        defects.append(Defect.floating)
    if "Isolator" in filename and "Partikel" in filename:
        defects.append(Defect.particle_insulator)
    elif "Partikel" in filename and "Isolator" not in filename:
        defects.append(Defect.free_particle)
    if len(defects) != 1:
        raise ValueError(f"No or multiple defects found: {filename}")
    return defects[0]


def _do_sanity_test(df: pd.DataFrame, filepath):
    if TIME_IN_FILE not in df or PD not in df:
        raise ValueError(f"TIME or PD column missing in file: {filepath}")

    if len(df.columns) > 5:
        raise ValueError(f"Unexpected columns in file: {filepath}")

    if (
        df[TIME_IN_FILE].min() < 0.0
        or not df[TIME_IN_FILE].equals(df[TIME_IN_FILE].sort_values())
        or not df[TIME_IN_FILE].is_unique
    ):
        raise ValueError(f"Time values are corrupt in file: {filepath}")


def read(
    filepath,
    treat_neg_values: TreatNegValues = TreatNegValues.nothing,
    labeled_file: bool = True,
) -> pd.DataFrame:
    experiment = pd.read_csv(
        filepath,
        sep=SEPERATOR,
        decimal=DECIMAL_SIGN,
        usecols=[TIME_IN_FILE, PD],
        dtype={PD: "float32"},
    )
    _do_sanity_test(experiment, filepath)

    assert TIME_UNIT == "ms"
    experiment[TIME_IN_FILE] *= 1000
    experiment[TIME_DIFF] = experiment[TIME_IN_FILE].diff()
    experiment.attrs[START_TIME] = experiment[TIME_IN_FILE].iloc[0]
    experiment.drop(columns=TIME_IN_FILE, inplace=True)

    if treat_neg_values is TreatNegValues.zero:
        experiment.loc[:, PD].clip(lower=0, inplace=True)
    elif treat_neg_values is TreatNegValues.absolute:
        experiment.loc[:, PD] = experiment[PD].abs()

    experiment.attrs[PATH] = str(filepath)
    filename = Path(filepath).stem
    if labeled_file:
        experiment.attrs[VOLTAGE_SIGN] = _get_voltage_sign(filename)
        experiment.attrs[CLASS] = get_defect(filename)

    return experiment.iloc[1:].reset_index(drop=True)


def read_recursive(
    dir_path, treat_neg_values: TreatNegValues = TreatNegValues.nothing
) -> Tuple[List[pd.DataFrame], list]:
    csv_filepaths = list(Path(dir_path).rglob("*.csv"))
    measurements = [read(f, treat_neg_values) for f in csv_filepaths]

    return measurements, csv_filepaths


def get_defects(measurements: List[pd.DataFrame]) -> List[Defect]:
    return [measurement.attrs[CLASS] for measurement in measurements]
