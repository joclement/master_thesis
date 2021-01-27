import numpy as np
import pandas as pd
import pytest

from thesis import data, models


def test_dt_seqfinger_ott_model(measurements):
    models_config = {
        "dt-seqfinger_ott": {
            "normalize": True,
            "data": {
                "duration": "30 seconds",
                "step_duration": "10 seconds",
            },
        }
    }
    y = pd.Series(data.get_defects(measurements))

    modelHandler = models.ModelHandler(measurements, y, models_config, False, None)
    pipeline, X = modelHandler.get_model_with_data("dt-seqfinger_ott")

    scaler = list(pipeline.named_steps.values())[0]
    X_transformed = scaler.fit_transform(X)
    assert X.shape == X_transformed.shape
    assert np.nanmax(X_transformed) == pytest.approx(1.0)
    assert np.nanmin(X_transformed) == 0.0
