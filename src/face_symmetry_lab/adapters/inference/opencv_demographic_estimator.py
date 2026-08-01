"""OpenCV DNN adapter for Open Model Zoo age/gender IR assets."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.models import DemographicEstimate

IMAGE_SIZE = (62, 62)
OUTPUT_NAMES = ["fc3_a", "prob"]


class DemographicInferenceError(RuntimeError):
    """Recoverable failure at the local demographic-model boundary."""


class _DnnNet(Protocol):
    def setInput(self, blob: NDArray[np.float32]) -> None: ...  # noqa: N802

    def forward(self, output_names: list[str]) -> list[NDArray[np.float32]]: ...


NetFactory = Callable[[Path, Path], _DnnNet]


def _default_net_factory(xml_path: Path, bin_path: Path) -> _DnnNet:
    try:
        import cv2

        return cast(_DnnNet, cv2.dnn.readNet(str(bin_path), str(xml_path)))
    except Exception as error:
        raise DemographicInferenceError("年齢・性別モデルを初期化できません。") from error


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_sha256(xml_path: Path, bin_path: Path) -> str:
    identity = f"xml:{_file_sha256(xml_path)}\nbin:{_file_sha256(bin_path)}".encode()
    return sha256(identity).hexdigest()


def preprocess_demographic_image(rgb_image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Resize RGB input and return the model's unscaled NCHW BGR blob."""
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise DemographicInferenceError("年齢・性別推定にはuint8のRGB画像が必要です。")
    resized = Image.fromarray(image, mode="RGB").resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )
    bgr = np.asarray(resized, dtype=np.float32)[..., ::-1]
    return np.ascontiguousarray(np.transpose(bgr, (2, 0, 1))[np.newaxis, ...])


class OpenCvDemographicEstimator:
    """Local CPU estimator for continuous age and binary appearance labels."""

    def __init__(
        self,
        *,
        xml_path: Path,
        bin_path: Path,
        model_version: str,
        net_factory: NetFactory = _default_net_factory,
    ) -> None:
        if not xml_path.is_file():
            raise DemographicInferenceError("年齢・性別モデルのXMLが見つかりません。")
        if not bin_path.is_file():
            raise DemographicInferenceError("年齢・性別モデルのBINが見つかりません。")
        self._model_version = model_version
        self._model_sha256 = _combined_sha256(xml_path, bin_path)
        self._net = net_factory(xml_path, bin_path)

    def estimate(self, rgb_image: NDArray[np.uint8]) -> DemographicEstimate:
        blob = preprocess_demographic_image(rgb_image)
        try:
            self._net.setInput(blob)
            age_output, gender_output = self._net.forward(OUTPUT_NAMES)
            age_values = np.asarray(age_output, dtype=np.float64).reshape(-1)
            probabilities = np.asarray(gender_output, dtype=np.float64).reshape(-1)
        except DemographicInferenceError:
            raise
        except Exception as error:
            raise DemographicInferenceError("年齢・性別推定を実行できませんでした。") from error
        if age_values.size != 1 or probabilities.size != 2:
            raise DemographicInferenceError("年齢・性別モデルの出力shapeが不正です。")
        try:
            return build_demographic_estimate(
                estimated_age_years=float(age_values[0]) * 100.0,
                gender_probabilities=(float(probabilities[0]), float(probabilities[1])),
                model_version=self._model_version,
                model_sha256=self._model_sha256,
                observed_at=datetime.now(UTC),
            )
        except ValueError as error:
            raise DemographicInferenceError("年齢・性別モデルの出力が不正です。") from error
