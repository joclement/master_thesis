import pandas as pd


def read(filepath):
    df = pd.read_csv(filepath, sep=";", decimal=",")
    assert df["Time [s]"][0] == 0
    df["Time [s]"] = pd.to_timedelta(df["Time [s]"], "seconds")
    df["TimeDiff [s]"] = df["Time [s]"].diff()
    df = df.drop(columns=["Time [s]"])
    return df
