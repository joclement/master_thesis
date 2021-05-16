from pathlib import Path

from thesis import illustrate_dtw


def test_main(tmpdir):
    filepath = Path(tmpdir, "illustration.png")
    assert not filepath.exists()
    illustrate_dtw.main(str(filepath))
    assert filepath.exists()
