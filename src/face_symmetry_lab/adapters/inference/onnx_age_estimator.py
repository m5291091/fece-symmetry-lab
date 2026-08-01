from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_symmetry_lab.domain.age import build_age_estimate
from face_symmetry_lab.domain.models import AGE_BANDS, AgeEstimate

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AgeInferenceError(RuntimeError):
    """Recoverable failure at the age-model boundary."""


class _InputMetadata(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_InputMetadata]: ...

    def run(
        self,
        output_names: Any,
        input_feed: dict[str, NDArray[np.float32]],
    ) -> list[Any]: ...


SessionFactory = Callable[[Path], _InferenceSession]


def _default_session_factory(model_path: Path) -> _InferenceSession:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise AgeInferenceError("ONNX Runtime を読み込めません。") from error

    try:
        return cast(
            _InferenceSession,
            ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            ),
        )
    except Exception as error:
        raise AgeInferenceError("年齢推定モデルを初期化できません。") from error


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_age_image(rgb_image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Match the ImageNet preprocessing used by the seminar notebook."""
    if rgb_image.dtype != np.uint8 or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise AgeInferenceError("年齢推定には uint8 の RGB 画像が必要です。")

    resized = Image.fromarray(rgb_image, mode="RGB").resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )
    normalized = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / (IMAGENET_STD)
    channels_first = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(channels_first[np.newaxis, ...], dtype=np.float32)


class OnnxAgeEstimator:
    """Local ONNX adapter for the eight-class MobileNetV2 age model."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        session_factory: SessionFactory = _default_session_factory,
    ) -> None:
        if not model_path.is_file():
            raise AgeInferenceError("年齢推定モデルが見つかりません。")
        self._model_version = model_version
        self._model_sha256 = _file_sha256(model_path)
        self._session = session_factory(model_path)
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise AgeInferenceError("年齢推定モデルの入力仕様が一致しません。")
        self._input_name = inputs[0].name

    def estimate(self, rgb_image: NDArray[np.uint8]) -> AgeEstimate:
        batch = preprocess_age_image(rgb_image)
        try:
            outputs = self._session.run(None, {self._input_name: batch})
            logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
        except AgeInferenceError:
            raise
        except Exception as error:
            raise AgeInferenceError("年齢推定を実行できませんでした。") from error

        if logits.size != len(AGE_BANDS):
            raise AgeInferenceError("年齢推定モデルは 8 クラスの出力が必要です。")

        try:
            return build_age_estimate(
                logits=tuple(float(value) for value in logits),
                model_version=self._model_version,
                model_sha256=self._model_sha256,
                observed_at=datetime.now(UTC),
            )
        except ValueError as error:
            raise AgeInferenceError("年齢推定モデルの出力が不正です。") from error
