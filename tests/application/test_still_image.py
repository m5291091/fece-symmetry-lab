import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import LandmarkDetectionError
from face_symmetry_lab.application.still_image import detect_still_image_landmarks
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)


class FakeImageLandmarker:
    def __init__(self, face_count: int = 1) -> None:
        self.face_count = face_count
        self.error: Exception | None = None
        self.calls = 0

    def detect(self, rgb_image: NDArray[np.uint8], *, frame_id: int = 0) -> LandmarkFrame:
        self.calls += 1
        if self.error is not None:
            raise self.error
        points = tuple((0.4 + 0.1 * i, 0.5) for i in range(3)) if self.face_count == 1 else ()
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=self.face_count,
            points=points,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
        )

    def close(self) -> None:
        return None


def _rgb(height: int = 40, width: int = 40) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_single_face_images_get_an_annotated_preview() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=1), _rgb())

    assert result.frame.status is TrackingStatus.SINGLE_FACE
    assert result.landmark_count == 3
    assert result.preview.sum() > 0
    assert result.preview.shape == (40, 40, 3)


def test_multi_face_images_are_reported_without_an_overlay() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=2), _rgb())

    assert result.frame.status is TrackingStatus.MULTI_FACE
    assert result.landmark_count == 0
    assert result.preview.sum() == 0


def test_no_face_images_are_reported_without_an_overlay() -> None:
    result = detect_still_image_landmarks(FakeImageLandmarker(face_count=0), _rgb())

    assert result.frame.status is TrackingStatus.NO_FACE
    assert result.preview.sum() == 0


def test_the_source_image_is_not_mutated() -> None:
    source = _rgb()

    detect_still_image_landmarks(FakeImageLandmarker(face_count=1), source)

    assert source.sum() == 0


def test_detection_failures_are_surfaced_to_the_caller() -> None:
    landmarker = FakeImageLandmarker()
    landmarker.error = LandmarkDetectionError("model failure")

    with pytest.raises(LandmarkDetectionError):
        detect_still_image_landmarks(landmarker, _rgb())


def test_still_images_are_stateless_and_reuse_frame_id_zero() -> None:
    landmarker = FakeImageLandmarker()

    first = detect_still_image_landmarks(landmarker, _rgb())
    second = detect_still_image_landmarks(landmarker, _rgb())

    assert first.frame.frame_id == 0
    assert second.frame.frame_id == 0
    assert landmarker.calls == 2


def test_still_image_midline_can_be_turned_on_and_off() -> None:
    source = _rgb()
    landmarker = FakeImageLandmarker()

    shown = detect_still_image_landmarks(
        landmarker,
        source,
        show_midline=True,
        midline_indices=(0, 1, 2),
    )
    hidden = detect_still_image_landmarks(
        landmarker,
        source,
        show_midline=False,
        midline_indices=(0, 1, 2),
    )

    assert not np.array_equal(shown.preview, hidden.preview)
