import pandas as pd

TIMEDIFF = "TimeDiff [s]"
PD = "A [nV]"
TEST_VOLTAGE = "Voltage [kV]"
TIME = "Time [s]"

SEPERATOR = ";"
DECIMAL_SIGN = ","


def read(filepath):
    df = pd.read_csv(filepath, sep=SEPERATOR, decimal=DECIMAL_SIGN)
    assert df[TIME][0] == 0
    df[TIMEDIFF] = df[TIME].diff()
    return df
