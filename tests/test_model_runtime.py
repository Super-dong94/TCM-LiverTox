from __future__ import annotations

from pathlib import Path

import numpy as np

from prediction_scripts.model_runtime import (
    load_model_bundles,
    predict_multimodel_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    ROOT / "models" / "01best_model_tuned.joblib",
    ROOT / "models" / "02best_model_tuned.joblib",
    ROOT / "models" / "03best_model_tuned.joblib",
)


def test_real_model_bundles_predict_urea() -> None:
    models = load_model_bundles(MODEL_PATHS)
    outputs = predict_multimodel_pipeline(["C(=O)(N)N"], models)

    assert len(outputs) == 6
    for index, values in enumerate(outputs):
        assert values.shape == (1,)
        if index % 2 == 0:
            assert int(values[0]) in {0, 1}
        else:
            assert np.isfinite(values[0])
            assert 0.0 <= float(values[0]) <= 1.0
