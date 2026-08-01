from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.domain.models import GenderLabel


class FakeNet:
    def __init__(self, age: float = 0.365, probabilities: tuple[float, float] = (0.2, 0.8)):
        self.age = age
        self.probabilities = probabilities
        self.last_blob: NDArray[np.float32] | None = None

    def setInput(self, blob: NDArray[np.float32]) -> None:  # noqa: N802
        self.last_blob = blob

    def forward(self, output_names: list[str]) -> list[NDArray[np.float32]]:
        assert output_names == ["fc3_a", "prob"]
        return [
            np.array([[[[self.age]]]], dtype=np.float32),
            np.array([[[[self.probabilities[0]]], [[self.probabilities[1]]]]], dtype=np.float32),
        ]


def _model_files(tmp_path: Path) -> tuple[Path, Path]:
    xml_path = tmp_path / "age-gender.xml"
    bin_path = tmp_path / "age-gender.bin"
    xml_path.write_bytes(b"test-graph")
    bin_path.write_bytes(b"test-weights")
    return xml_path, bin_path


def test_estimator_uses_bgr_62_square_and_maps_continuous_outputs(tmp_path: Path) -> None:
    xml_path, bin_path = _model_files(tmp_path)
    net = FakeNet()
    estimator = OpenCvDemographicEstimator(
        xml_path=xml_path,
        bin_path=bin_path,
        model_version="openvino-age-gender/1.0.0",
        net_factory=lambda _xml, _bin: net,
    )
    image = np.zeros((32, 24, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 2] = 30

    estimate = estimator.estimate(image)

    assert estimate.estimated_age_years == pytest.approx(36.5)
    assert estimate.gender is GenderLabel.MALE
    assert len(estimate.model_sha256) == 64
    assert net.last_blob is not None
    assert net.last_blob.shape == (1, 3, 62, 62)
    assert net.last_blob.dtype == np.float32
    assert net.last_blob[0, 0, 0, 0] == pytest.approx(30)  # RGB -> BGR
    assert net.last_blob[0, 2, 0, 0] == pytest.approx(10)


@pytest.mark.parametrize("age", [0.17, 0.76, float("nan")])
def test_estimator_rejects_age_outside_supported_range(tmp_path: Path, age: float) -> None:
    xml_path, bin_path = _model_files(tmp_path)
    estimator = OpenCvDemographicEstimator(
        xml_path=xml_path,
        bin_path=bin_path,
        model_version="openvino-age-gender/1.0.0",
        net_factory=lambda _xml, _bin: FakeNet(age=age),
    )

    with pytest.raises(DemographicInferenceError):
        estimator.estimate(np.zeros((32, 32, 3), dtype=np.uint8))


def test_estimator_requires_both_ir_files(tmp_path: Path) -> None:
    xml_path, _ = _model_files(tmp_path)

    with pytest.raises(DemographicInferenceError, match="BIN"):
        OpenCvDemographicEstimator(
            xml_path=xml_path,
            bin_path=tmp_path / "missing.bin",
            model_version="openvino-age-gender/1.0.0",
        )
