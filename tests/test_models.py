import pandas as pd
import pytest

from thesis import data, models


@pytest.fixture
def models_config():
    return {
        "knn_dtw-seqfinger_seqown": {
            "normalize": True,
            "data": {
                "duration": "30 seconds",
                "step_duration": "10 seconds",
            },
        }
    }


def test_knn_dtw_seqfinger_seqown_model(models_config, measurements):
    y = pd.Series(data.get_defects(measurements))

    modelHandler = models.ModelHandler(y, models_config, False, None)
    pipeline = modelHandler.get_model("knn_dtw-seqfinger_seqown")

    assert len(pipeline.steps) == 3
    pipeline.fit(measurements, y)
