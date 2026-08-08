import math
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
    MediaPipeImageLandmarker,
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


class FakeImageDetector:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self.faces = faces
        self.calls = 0
        self.closed = False
        self.error: Exception | None = None

    def detect(self, image: Any) -> SimpleNamespace:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(face_landmarks=self.faces)

    def close(self) -> None:
        self.closed = True


def _image_landmarker(tmp_path: Path, detector: FakeImageDetector) -> MediaPipeImageLandmarker:
    return MediaPipeImageLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )


def test_image_mode_returns_the_single_faces_landmarks(tmp_path: Path) -> None:
    detector = FakeImageDetector([_face((0.4, 0.6), (0.6, 0.6))])
    landmarker = _image_landmarker(tmp_path, detector)

    frame = landmarker.detect(_rgb())

    assert frame.face_count == 1
    assert frame.points == ((0.4, 0.6), (0.6, 0.6))
    assert detector.calls == 1


def test_image_mode_reports_no_face(tmp_path: Path) -> None:
    landmarker = _image_landmarker(tmp_path, FakeImageDetector([]))

    assert landmarker.detect(_rgb()).face_count == 0


def test_image_mode_does_not_pick_a_face_when_several_are_present(tmp_path: Path) -> None:
    detector = FakeImageDetector([_face((0.2, 0.2)), _face((0.8, 0.8))])
    landmarker = _image_landmarker(tmp_path, detector)

    frame = landmarker.detect(_rgb())

    assert frame.face_count == 2
    assert frame.points == ()


def test_image_mode_rejects_non_rgb_input(tmp_path: Path) -> None:
    landmarker = _image_landmarker(tmp_path, FakeImageDetector([]))

    with pytest.raises(LandmarkDetectionError, match="RGB"):
        landmarker.detect(np.zeros((8, 8), dtype=np.uint8))


def test_image_mode_wraps_detector_failures(tmp_path: Path) -> None:
    detector = FakeImageDetector([])
    detector.error = RuntimeError("graph failure")
    landmarker = _image_landmarker(tmp_path, detector)

    with pytest.raises(LandmarkDetectionError, match="ランドマーク"):
        landmarker.detect(_rgb())


def test_image_mode_requires_an_existing_model(tmp_path: Path) -> None:
    with pytest.raises(LandmarkDetectionError, match="モデル"):
        MediaPipeImageLandmarker(
            model_path=tmp_path / "missing.task",
            model_version="face-landmarker/1.0.0",
            detector_factory=lambda _: FakeImageDetector([]),
        )


def test_image_mode_close_releases_the_detector(tmp_path: Path) -> None:
    detector = FakeImageDetector([])
    landmarker = _image_landmarker(tmp_path, detector)

    landmarker.close()

    assert detector.closed is True
    with pytest.raises(LandmarkDetectionError, match="停止"):
        landmarker.detect(_rgb())


def test_video_and_image_landmarkers_are_separate_instances(tmp_path: Path) -> None:
    """§14.3: VIDEO のインスタンスを静止画へ流用してはならない。"""
    model_path = _model_file(tmp_path)
    video_detector = FakeDetector([])
    image_detector = FakeImageDetector([])

    video = MediaPipeFaceLandmarker(
        model_path=model_path,
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: video_detector,
        image_factory=lambda rgb: rgb,
    )
    image = MediaPipeImageLandmarker(
        model_path=model_path,
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: image_detector,
        image_factory=lambda rgb: rgb,
    )

    video.close()

    assert video_detector.closed is True
    assert image_detector.closed is False
    assert image.model_sha256 == video.model_sha256


def _rotation_matrix_y(degrees: float) -> list[list[float]]:
    angle = math.radians(degrees)
    return [
        [math.cos(angle), 0.0, math.sin(angle), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class FakePoseDetector(FakeImageDetector):
    def __init__(self, faces, matrices) -> None:
        super().__init__(faces)
        self.matrices = matrices

    def detect(self, image: Any) -> SimpleNamespace:
        super().detect(image)
        return SimpleNamespace(
            face_landmarks=self.faces,
            facial_transformation_matrixes=self.matrices,
        )


def test_the_head_pose_is_read_from_the_transformation_matrix(tmp_path: Path) -> None:
    detector = FakePoseDetector([_face((0.5, 0.5))], [_rotation_matrix_y(12.0)])
    landmarker = MediaPipeImageLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )

    frame = landmarker.detect(_rgb())

    assert frame.pose is not None
    assert frame.pose.yaw_deg == pytest.approx(12.0, abs=1e-6)


def test_an_unusable_pose_matrix_leaves_the_pose_empty(tmp_path: Path) -> None:
    detector = FakePoseDetector([_face((0.5, 0.5))], [[[0.0] * 4] * 4])
    landmarker = MediaPipeImageLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )

    frame = landmarker.detect(_rgb())

    assert frame.face_count == 1
    assert frame.pose is None


def test_multi_face_results_carry_no_pose(tmp_path: Path) -> None:
    detector = FakePoseDetector(
        [_face((0.2, 0.2)), _face((0.8, 0.8))],
        [_rotation_matrix_y(5.0), _rotation_matrix_y(-5.0)],
    )
    landmarker = MediaPipeImageLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )

    assert landmarker.detect(_rgb()).pose is None


# --------------------------------------------------------------------------
# 推定奥行き z（§8.1.1, FR-009）
# --------------------------------------------------------------------------


def _face_3d(*points: tuple[float, float, float]) -> list[SimpleNamespace]:
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


def test_the_depth_reaches_the_landmark_frame(tmp_path: Path) -> None:
    detector = FakeDetector([_face_3d((0.25, 0.5, -0.02), (0.75, 0.5, 0.03))])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)

    assert frame.points == ((0.25, 0.5), (0.75, 0.5))
    assert frame.depths == (-0.02, 0.03)
    assert frame.has_depth is True


def test_a_result_without_z_leaves_the_depth_missing_and_the_2d_path_intact(
    tmp_path: Path,
) -> None:
    """z を返さない検出器でも、xy はこれまでどおり届く（不変条件 11, 12）。"""
    detector = FakeDetector([[SimpleNamespace(x=0.25, y=0.5), SimpleNamespace(x=0.75, y=0.5)]])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)

    assert frame.face_count == 1
    assert frame.points == ((0.25, 0.5), (0.75, 0.5))
    assert frame.depths is None
    assert frame.has_depth is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_z_makes_the_whole_frames_depth_missing(tmp_path: Path, bad: float) -> None:
    """z の欠測は欠測として扱い、0 で埋めず、xy 経路も止めない。"""
    detector = FakeDetector([_face_3d((0.25, 0.5, -0.02), (0.75, 0.5, bad))])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)

    assert frame.points == ((0.25, 0.5), (0.75, 0.5))
    assert frame.depths is None


def test_multi_face_results_carry_no_depth(tmp_path: Path) -> None:
    detector = FakeDetector([_face_3d((0.2, 0.2, 0.0)), _face_3d((0.8, 0.8, 0.0))])
    landmarker = _landmarker(tmp_path, detector)

    frame = landmarker.detect_video(_rgb(), frame_id=1, timestamp_ms=0)

    assert frame.face_count == 2
    assert frame.depths is None


def test_image_mode_also_carries_the_depth(tmp_path: Path) -> None:
    detector = FakeImageDetector([_face_3d((0.4, 0.6, 0.05), (0.6, 0.6, -0.05))])
    landmarker = _image_landmarker(tmp_path, detector)

    frame = landmarker.detect(_rgb())

    assert frame.depths == (0.05, -0.05)


def test_image_mode_treats_a_missing_z_as_missing(tmp_path: Path) -> None:
    detector = FakeImageDetector([[SimpleNamespace(x=0.4, y=0.6)]])
    landmarker = _image_landmarker(tmp_path, detector)

    frame = landmarker.detect(_rgb())

    assert frame.points == ((0.4, 0.6),)
    assert frame.depths is None


def test_the_frame_records_the_image_size_for_pixel_conversion(tmp_path: Path) -> None:
    detector = FakePoseDetector([_face((0.5, 0.5))], [_rotation_matrix_y(0.0)])
    landmarker = MediaPipeImageLandmarker(
        model_path=_model_file(tmp_path),
        model_version="face-landmarker/1.0.0",
        detector_factory=lambda _: detector,
        image_factory=lambda rgb: rgb,
    )

    frame = landmarker.detect(_rgb(height=48, width=64))

    assert (frame.width, frame.height) == (64, 48)
