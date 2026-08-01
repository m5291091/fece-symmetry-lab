from __future__ import annotations

import numpy as np
import pytest

from face_symmetry_lab.adapters.io.model_store import InstallStatus
from face_symmetry_lab.ui.model_setup import (
    configured_demographic_path,
    demographic_model_status,
)


@pytest.mark.model
def test_pinned_demographic_onnx_loads_and_exposes_expected_outputs() -> None:
    onnx_path = configured_demographic_path()
    if demographic_model_status(onnx_path) is not InstallStatus.READY:
        pytest.skip("manifest検証済みの年齢・性別モデルが未配置です。")

    import cv2

    if not hasattr(cv2, "dnn"):
        pytest.skip("cv2 DNN runtimeが現在の環境でロードされていません。")

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(np.zeros((1, 3, 62, 62), dtype=np.float32))
    age_output, gender_output = net.forward(["fc3_a", "prob"])

    assert np.asarray(age_output).shape == (1, 1, 1, 1)
    assert np.asarray(gender_output).shape == (1, 2, 1, 1)
    assert np.isfinite(age_output).all()
    assert np.isfinite(gender_output).all()
