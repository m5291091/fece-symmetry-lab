from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    FrameOrderError,
    LandmarkDetectionError,
    MediaPipeFaceLandmarker,
)


class FakeDetector:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self.faces = faces
        self.timestamps: list[int] = []
        self.closed = False
        self.error: Exception | None = None

    def detect_for_video(self, image: Any, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(face_landmarks=self.faces)

    def close(self) -> None:
        self.closed = True


def _face(*points: tuple[float, float]) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


def _model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "face_landmarker.task"
    model_path.write_bytes(b"test-landmarker")
    return model_path


def _landmarker(tmp_path: Path, detector: FakeDetector) -> MediaPipeFaceLandmarker:
    return MediaPipeFaceLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )


def _rgb(height: int = 32, width: int = 32) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_single_face_result_is_converted_to_a_landmark_frame(tmp_path: Path) -> None:
    detector = FakeDetector([_face((0.25, 0.5), (0.75, 0.5))])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)

    assert frame.frame_id == 1
    assert frame.face_count == 1
    assert frame.points == ((0.25, 0.5), (0.75, 0.5))
    assert detector.timestamps == [0]


def test_no_face_result_reports_zero_faces(tmp_path: Path) -> None:
    landmarker = _landmarker(tmp_path, FakeDetector([]))

    frame = landmarker.detect_video(_rgb(), frame_id=2, timestamp_ms=0)

    assert frame.face_count == 0
    assert frame.points == ()


def test_multiple_faces_are_reported_without_picking_one(tmp_path: Path) -> None:
    detector = FakeDetector([_face((0.2, 0.2)), _face((0.8, 0.8))])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=3, timestamp_ms=0)

    assert frame.face_count == 2
    assert frame.points == ()


def test_timestamps_must_increase_monotonically(tmp_path: Path) -> None:
    landmarker = _landmarker(tmp_path, FakeDetector([_face((0.5, 0.5))]))
    landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=100)

    with pytest.raises(FrameOrderError):
        landmarker.detect_video(_rgb(), frame_id=2, timestamp_ms=100)


def test_out_of_order_frames_do_not_reach_the_detector(tmp_path: Path) -> None:
    detector = FakeDetector([_face((0.5, 0.5))])
    landmarker = _landmarker(tmp_path, detector)
    landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=100)

    with pytest.raises(FrameOrderError):
        landmarker.detect_video(_rgb(), frame_id=2, timestamp_ms=40)

    assert detector.timestamps == [100]


def test_detector_failures_become_recoverable_adapter_errors(tmp_path: Path) -> None:
    detector = FakeDetector([_face((0.5, 0.5))])
    detector.error = RuntimeError("graph failure")
    landmarker = _landmarker(tmp_path, detector)

    with pytest.raises(LandmarkDetectionError, match="ランドマーク"):
        landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)


def test_non_finite_landmarks_are_rejected(tmp_path: Path) -> None:
    detector = FakeDetector([_face((0.5, float("nan")))])
    landmarker = _landmarker(tmp_path, detector)

    with pytest.raises(LandmarkDetectionError):
        landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)


def test_non_rgb_input_is_rejected(tmp_path: Path) -> None:
    landmarker = _landmarker(tmp_path, FakeDetector([]))

    with pytest.raises(LandmarkDetectionError, match="RGB"):
        landmarker.detect_video(np.zeros((8, 8), dtype=np.uint8), frame_id=1, timestamp_ms=0)


def test_model_identity_is_tracked_by_version_and_hash(tmp_path: Path) -> None:
    landmarker = _landmarker(tmp_path, FakeDetector([]))

    assert landmarker.model_version == "face-landmarker/1.0.0"
    assert len(landmarker.model_sha256) == 64


def test_missing_model_file_is_reported_before_any_detection(tmp_path: Path) -> None:
    with pytest.raises(LandmarkDetectionError, match="モデル"):
        MediaPipeFaceLandmarker(
            model_path=tmp_path / "missing.task",
            model_version="face-landmarker/1.0.0",
            detector_factory=lambda _: FakeDetector([]),
        )


def test_close_releases_the_detector(tmp_path: Path) -> None:
    detector = FakeDetector([])
    landmarker = _landmarker(tmp_path, detector)

    landmarker.close()

    assert detector.closed is True


def test_detection_after_close_is_refused(tmp_path: Path) -> None:
    landmarker = _landmarker(tmp_path, FakeDetector([]))
    landmarker.close()

    with pytest.raises(LandmarkDetectionError, match="停止"):
        landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)
