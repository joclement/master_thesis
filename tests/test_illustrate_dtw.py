from pathlib import Path

from jc_thesis_code import illustrate_dtw


def test_main(tmpdir):
    filepath = Path(tmpdir, "illustration.png")
    assert not filepath.exists()
    illustrate_dtw.main(str(filepath))
    assert filepath.exists()
