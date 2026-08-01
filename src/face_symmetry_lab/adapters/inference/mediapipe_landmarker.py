"""MediaPipe Face Landmarker adapter for realtime video.

The adapter owns the MediaPipe boundary only: it converts an RGB array into a
MediaPipe image, runs ``detect_for_video`` with a strictly increasing
timestamp, and returns an immutable :class:`LandmarkFrame`. It never scores,
never picks a face when several are present, and never fetches a model.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    build_landmark_frame,
)

#: Detecting two faces is enough to report MULTI_FACE without ranking them.
MAX_TRACKED_FACES = 2


class LandmarkDetectionError(RuntimeError):
    """Recoverable failure at the landmark model boundary."""


class FrameOrderError(LandmarkDetectionError):
    """A frame arrived with a non-increasing timestamp (§14.4)."""


class _Detector(Protocol):
    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...

    def close(self) -> None: ...


DetectorFactory = Callable[[Path], _Detector]
ImageFactory = Callable[[NDArray[np.uint8]], Any]


def _default_detector_factory(model_path: Path) -> _Detector:
    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )
    except ImportError as error:
        raise LandmarkDetectionError("MediaPipe を読み込めません。") from error

    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=MAX_TRACKED_FACES,
        )
        detector: _Detector = FaceLandmarker.create_from_options(options)
    except Exception as error:
        raise LandmarkDetectionError("ランドマークモデルを初期化できません。") from error
    return detector


def _default_image_factory(rgb_image: NDArray[np.uint8]) -> Any:
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MediaPipeFaceLandmarker:
    """Local Face Landmarker adapter running in ``RunningMode.VIDEO``."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        detector_factory: DetectorFactory = _default_detector_factory,
        image_factory: ImageFactory = _default_image_factory,
    ) -> None:
        if not model_path.is_file():
            raise LandmarkDetectionError("ランドマークモデルが見つかりません。")
        self._model_version = model_version
        self._model_sha256 = _file_sha256(model_path)
        self._image_factory = image_factory
        self._detector: _Detector | None = detector_factory(model_path)
        self._last_timestamp_ms = -1

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        detector = self._detector
        if detector is None:
            raise LandmarkDetectionError("ランドマーク推論は停止済みです。")
        if rgb_image.dtype != np.uint8 or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise LandmarkDetectionError("ランドマーク推論には uint8 の RGB 画像が必要です。")
        if timestamp_ms <= self._last_timestamp_ms:
            raise FrameOrderError("フレームの時刻が単調増加していません。")

        self._last_timestamp_ms = timestamp_ms
        try:
            result = detector.detect_for_video(self._image_factory(rgb_image), timestamp_ms)
        except Exception as error:
            raise LandmarkDetectionError("ランドマークを検出できませんでした。") from error

        faces = list(getattr(result, "face_landmarks", []) or [])
        points: tuple[tuple[float, float], ...] = ()
        if len(faces) == 1:
            points = tuple((float(point.x), float(point.y)) for point in faces[0])

        try:
            return build_landmark_frame(
                frame_id=frame_id,
                face_count=len(faces),
                points=points,
            )
        except LandmarkError as error:
            raise LandmarkDetectionError("ランドマーク出力が不正です。") from error

    def close(self) -> None:
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()
