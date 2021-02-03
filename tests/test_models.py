import numpy as np
import pandas as pd
import pytest

from thesis import data, models


def test_knn_dtw_seqfinger_seqown_model(measurements):
    models_config = {
        "knn_dtw-seqfinger_seqown": {
            "normalize": True,
            "data": {
                "duration": "30 seconds",
                "step_duration": "10 seconds",
            },
        }
    }
    y = pd.Series(data.get_defects(measurements))

    modelHandler = models.ModelHandler(measurements, y, models_config, False, None)
    pipeline, X = modelHandler.get_model_with_data("knn_dtw-seqfinger_seqown")

    scaler = list(pipeline.named_steps.values())[0]
    X_transformed = scaler.fit_transform(X)
    assert X.shape == X_transformed.shape
    assert np.nanmax(X_transformed) == pytest.approx(1.0)
    assert np.nanmin(X_transformed) == 0.0
