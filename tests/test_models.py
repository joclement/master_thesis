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

    modelHandler = models.ModelHandler(set(y), models_config, False)
    pipeline = modelHandler.get_model("knn_dtw-seqfinger_seqown")

    assert len(pipeline.steps) == 3
    pipeline.fit(measurements, y)


def test_is_model_finger():
    with pytest.raises(ValueError):
        models.is_model_finger("Invalid input")

    assert not models.is_model_finger("knn_dtw-seqfinger_seqown")
    assert not models.is_model_finger("svm_dtw-seqfinger_seqown")
    assert not models.is_model_finger("knn-finger")

    assert models.is_model_finger("knn_dtw-finger_seqown")
    assert models.is_model_finger("knn_dtw-finger_ott")
    assert models.is_model_finger("knn-finger_ott")
