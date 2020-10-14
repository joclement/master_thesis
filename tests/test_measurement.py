from pathlib import Path

from thesis import measurement


def test_measurement_read(csv_filepath):
    df = measurement.read(csv_filepath)
    assert measurement.CLASS in df
    assert df[measurement.CLASS][0] == measurement.Defect.free_particle
    assert len(set(df[measurement.CLASS])) == 1
    assert measurement.VOLTAGE_SIGN in df
    assert df[measurement.VOLTAGE_SIGN][0] == measurement.VoltageSign.positive
    assert len(set(df[measurement.VOLTAGE_SIGN])) == 1


def test_measurement_read_recursive(csv_filepath):
    result = measurement.read_recursive(Path(csv_filepath).parent)
    assert len(result) == 1
