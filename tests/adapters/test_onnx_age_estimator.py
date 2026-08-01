from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.onnx_age_estimator import (
    AgeInferenceError,
    OnnxAgeEstimator,
)
from face_symmetry_lab.domain.models import AgeBand


class FakeSession:
    def __init__(self, output: NDArray[np.float32]) -> None:
        self.output = output
        self.last_feed: dict[str, NDArray[np.float32]] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input")]

    def run(
        self,
        output_names: Any,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[NDArray[np.float32]]:
        self.last_feed = input_feed
        return [self.output]


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "age.onnx"
    model_path.write_bytes(b"test-model")
    return model_path


def test_estimator_preprocesses_rgb_image_and_maps_logits(tmp_path: Path) -> None:
    session = FakeSession(np.array([[0, 0, 5, 0, 0, 0, 0, 0]], dtype=np.float32))
    estimator = OnnxAgeEstimator(
        model_path=_model_file(tmp_path),
        model_version="age-mobilenetv2-utkface/1.0.0",
        session_factory=lambda _: session,
    )
    image = np.full((32, 24, 3), 127, dtype=np.uint8)

    estimate = estimator.estimate(image)

    assert estimate.band is AgeBand.AGE_20_29
    assert len(estimate.model_sha256) == 64
    assert session.last_feed is not None
    batch = session.last_feed["input"]
    assert batch.shape == (1, 3, 224, 224)
    assert batch.dtype == np.float32


def test_estimator_rejects_non_rgb_input(tmp_path: Path) -> None:
    estimator = OnnxAgeEstimator(
        model_path=_model_file(tmp_path),
        model_version="age-mobilenetv2-utkface/1.0.0",
        session_factory=lambda _: FakeSession(np.zeros((1, 8), dtype=np.float32)),
    )

    with pytest.raises(AgeInferenceError, match="RGB"):
        estimator.estimate(np.zeros((32, 32), dtype=np.uint8))


def test_estimator_rejects_unexpected_model_output(tmp_path: Path) -> None:
    estimator = OnnxAgeEstimator(
        model_path=_model_file(tmp_path),
        model_version="age-mobilenetv2-utkface/1.0.0",
        session_factory=lambda _: FakeSession(np.zeros((1, 7), dtype=np.float32)),
    )

    with pytest.raises(AgeInferenceError, match="8 クラス"):
        estimator.estimate(np.zeros((32, 32, 3), dtype=np.uint8))


def test_estimator_requires_existing_model_file(tmp_path: Path) -> None:
    with pytest.raises(AgeInferenceError, match="モデル"):
        OnnxAgeEstimator(
            model_path=tmp_path / "missing.onnx",
            model_version="age-mobilenetv2-utkface/1.0.0",
        )
