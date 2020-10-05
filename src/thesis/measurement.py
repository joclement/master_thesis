import pandas as pd

TIMEDIFF = "TimeDiff [s]"
PD = "A [nV]"
TEST_VOLTAGE = "Voltage [kV]"
TIME = "Time [s]"


def read(filepath):
    df = pd.read_csv(filepath, sep=";", decimal=",")
    assert df[TIME][0] == 0
    df[TIMEDIFF] = df[TIME].diff()
    return df
