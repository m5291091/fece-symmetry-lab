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
from face_symmetry_lab.domain.pose import FacePose, PoseError, pose_from_matrix

#: Detecting two faces is enough to report MULTI_FACE without ranking them.
MAX_TRACKED_FACES = 2


class LandmarkDetectionError(RuntimeError):
    """Recoverable failure at the landmark model boundary."""


class FrameOrderError(LandmarkDetectionError):
    """A frame arrived with a non-increasing timestamp (§14.4)."""


class _Detector(Protocol):
    def close(self) -> None: ...


class _VideoDetector(_Detector, Protocol):
    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...


class _ImageDetector(_Detector, Protocol):
    def detect(self, image: Any) -> Any: ...


DetectorFactory = Callable[[Path], Any]
ImageFactory = Callable[[NDArray[np.uint8]], Any]


def _create_detector(model_path: Path, running_mode_name: str) -> Any:
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
            running_mode=getattr(RunningMode, running_mode_name),
            num_faces=MAX_TRACKED_FACES,
            output_facial_transformation_matrixes=True,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception as error:
        raise LandmarkDetectionError("ランドマークモデルを初期化できません。") from error


def _default_detector_factory(model_path: Path) -> Any:
    return _create_detector(model_path, "VIDEO")


def _default_image_detector_factory(model_path: Path) -> Any:
    # §14.3: still images need their own IMAGE-mode instance.
    return _create_detector(model_path, "IMAGE")


def _default_image_factory(rgb_image: NDArray[np.uint8]) -> Any:
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_pose(result: Any) -> FacePose | None:
    """Read the head pose, or None when the matrix is unusable (§8.2.1)."""
    matrices = list(getattr(result, "facial_transformation_matrixes", []) or [])
    if len(matrices) != 1:
        return None
    try:
        return pose_from_matrix(np.asarray(matrices[0], dtype=np.float64))
    except PoseError:
        # A missing pose becomes LOW_CONFIDENCE at the gate, not a dead frame.
        return None


def _to_landmark_frame(result: Any, frame_id: int, width: int, height: int) -> LandmarkFrame:
    """Convert a MediaPipe result without ever ranking or picking a face."""
    faces = list(getattr(result, "face_landmarks", []) or [])
    points: tuple[tuple[float, float], ...] = ()
    pose: FacePose | None = None
    if len(faces) == 1:
        points = tuple((float(point.x), float(point.y)) for point in faces[0])
        pose = _extract_pose(result)

    try:
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=len(faces),
            points=points,
            pose=pose,
            width=width,
            height=height,
        )
    except LandmarkError as error:
        raise LandmarkDetectionError("ランドマーク出力が不正です。") from error


class _LandmarkerBase:
    """Shared model identity and lifecycle for both running modes."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        detector_factory: DetectorFactory,
        image_factory: ImageFactory,
    ) -> None:
        if not model_path.is_file():
            raise LandmarkDetectionError("ランドマークモデルが見つかりません。")
        self._model_version = model_version
        self._model_sha256 = _file_sha256(model_path)
        self._image_factory = image_factory
        self._detector: Any | None = detector_factory(model_path)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def _active_detector(self, rgb_image: NDArray[np.uint8]) -> Any:
        detector = self._detector
        if detector is None:
            raise LandmarkDetectionError("ランドマーク推論は停止済みです。")
        if rgb_image.dtype != np.uint8 or rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise LandmarkDetectionError("ランドマーク推論には uint8 の RGB 画像が必要です。")
        return detector

    def close(self) -> None:
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()


class MediaPipeFaceLandmarker(_LandmarkerBase):
    """Local Face Landmarker adapter running in ``RunningMode.VIDEO``."""

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        detector_factory: DetectorFactory = _default_detector_factory,
        image_factory: ImageFactory = _default_image_factory,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_version=model_version,
            detector_factory=detector_factory,
            image_factory=image_factory,
        )
        self._last_timestamp_ms = -1

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        detector: _VideoDetector = self._active_detector(rgb_image)
        if timestamp_ms <= self._last_timestamp_ms:
            raise FrameOrderError("フレームの時刻が単調増加していません。")

        self._last_timestamp_ms = timestamp_ms
        try:
            result = detector.detect_for_video(self._image_factory(rgb_image), timestamp_ms)
        except Exception as error:
            raise LandmarkDetectionError("ランドマークを検出できませんでした。") from error

        height, width = rgb_image.shape[:2]
        return _to_landmark_frame(result, frame_id, int(width), int(height))


class MediaPipeImageLandmarker(_LandmarkerBase):
    """Local Face Landmarker adapter running in ``RunningMode.IMAGE``.

    Still images are stateless: no timestamp, no tracking history, and a
    separate instance from the video landmarker (§14.3).
    """

    def __init__(
        self,
        *,
        model_path: Path,
        model_version: str,
        detector_factory: DetectorFactory = _default_image_detector_factory,
        image_factory: ImageFactory = _default_image_factory,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_version=model_version,
            detector_factory=detector_factory,
            image_factory=image_factory,
        )

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame:
        detector: _ImageDetector = self._active_detector(rgb_image)
        try:
            result = detector.detect(self._image_factory(rgb_image))
        except Exception as error:
            raise LandmarkDetectionError("ランドマークを検出できませんでした。") from error

        height, width = rgb_image.shape[:2]
        return _to_landmark_frame(result, frame_id, int(width), int(height))
