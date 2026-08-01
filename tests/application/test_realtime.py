import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    FrameOrderError,
    LandmarkDetectionError,
)
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)


class FakeLandmarker:
    def __init__(self, face_count: int = 1) -> None:
        self.face_count = face_count
        self.error: Exception | None = None
        self.calls: list[tuple[int, int]] = []
        self.closed = False

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.calls.append((frame_id, timestamp_ms))
        if self.error is not None:
            raise self.error
        points = ((0.5, 0.5),) if self.face_count == 1 else ()
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=self.face_count,
            points=points,
        )

    def close(self) -> None:
        self.closed = True


class StepClock:
    """Monotonic clock stub advancing by a fixed step per call."""

    def __init__(self, step: float = 0.05) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def _rgb(height: int = 20, width: int = 20) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _pipeline(landmarker: FakeLandmarker, **kwargs: object) -> RealtimeLandmarkPipeline:
    return RealtimeLandmarkPipeline(
        landmarker=landmarker,
        clock=StepClock(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_single_face_frames_are_annotated_with_landmarks() -> None:
    pipeline = _pipeline(FakeLandmarker(face_count=1))

    annotated = pipeline.process(_rgb())

    assert annotated.shape == (20, 20, 3)
    assert annotated.sum() > 0
    assert pipeline.snapshot().status is TrackingStatus.SINGLE_FACE


def test_multi_face_frames_are_not_annotated() -> None:
    pipeline = _pipeline(FakeLandmarker(face_count=2))

    annotated = pipeline.process(_rgb())

    assert annotated.sum() == 0
    snapshot = pipeline.snapshot()
    assert snapshot.status is TrackingStatus.MULTI_FACE
    assert snapshot.face_count == 2
    assert snapshot.landmark_count == 0


def test_no_face_frames_report_the_tracking_status() -> None:
    pipeline = _pipeline(FakeLandmarker(face_count=0))

    pipeline.process(_rgb())

    assert pipeline.snapshot().status is TrackingStatus.NO_FACE


def test_frame_ids_and_timestamps_increase_monotonically() -> None:
    landmarker = FakeLandmarker()
    pipeline = _pipeline(landmarker)

    for _ in range(3):
        pipeline.process(_rgb())

    frame_ids = [call[0] for call in landmarker.calls]
    timestamps = [call[1] for call in landmarker.calls]
    assert frame_ids == [1, 2, 3]
    assert timestamps == sorted(set(timestamps))


def test_timestamps_stay_strictly_increasing_under_a_stalled_clock() -> None:
    landmarker = FakeLandmarker()
    pipeline = RealtimeLandmarkPipeline(landmarker=landmarker, clock=lambda: 1.0)

    pipeline.process(_rgb())
    pipeline.process(_rgb())

    timestamps = [call[1] for call in landmarker.calls]
    assert timestamps[1] > timestamps[0]


def test_a_busy_analyzer_returns_the_raw_frame_instead_of_queueing() -> None:
    landmarker = FakeLandmarker()
    pipeline = _pipeline(landmarker)

    with pipeline.scheduler.slot():
        annotated = pipeline.process(_rgb())

    assert annotated.sum() == 0
    assert landmarker.calls == []
    assert pipeline.snapshot().dropped_frames == 1


def test_detection_failures_keep_the_stream_running() -> None:
    landmarker = FakeLandmarker()
    landmarker.error = LandmarkDetectionError("boom")
    pipeline = _pipeline(landmarker)

    annotated = pipeline.process(_rgb())

    assert annotated.shape == (20, 20, 3)
    snapshot = pipeline.snapshot()
    assert snapshot.status is TrackingStatus.ERROR
    assert snapshot.consecutive_errors == 1


def test_consecutive_failures_are_counted_and_reset_on_recovery() -> None:
    landmarker = FakeLandmarker()
    landmarker.error = LandmarkDetectionError("boom")
    pipeline = _pipeline(landmarker)

    pipeline.process(_rgb())
    pipeline.process(_rgb())
    assert pipeline.snapshot().consecutive_errors == 2

    landmarker.error = None
    pipeline.process(_rgb())
    assert pipeline.snapshot().consecutive_errors == 0


def test_out_of_order_frames_are_skipped_without_raising() -> None:
    landmarker = FakeLandmarker()
    landmarker.error = FrameOrderError("out of order")
    pipeline = _pipeline(landmarker)

    annotated = pipeline.process(_rgb())

    assert annotated.sum() == 0
    assert pipeline.snapshot().status is TrackingStatus.ERROR


def test_unexpected_adapter_exceptions_are_contained() -> None:
    landmarker = FakeLandmarker()
    landmarker.error = ValueError("unexpected")
    pipeline = _pipeline(landmarker)

    annotated = pipeline.process(_rgb())

    assert annotated.shape == (20, 20, 3)
    assert pipeline.snapshot().status is TrackingStatus.ERROR


def test_snapshot_is_immutable() -> None:
    pipeline = _pipeline(FakeLandmarker())
    pipeline.process(_rgb())

    snapshot = pipeline.snapshot()

    with pytest.raises((AttributeError, TypeError)):
        snapshot.face_count = 99  # type: ignore[misc]


def test_close_releases_the_landmarker_and_stops_processing() -> None:
    landmarker = FakeLandmarker()
    pipeline = _pipeline(landmarker)

    pipeline.close()
    pipeline.process(_rgb())

    assert landmarker.closed is True
    assert landmarker.calls == []


def test_non_rgb_frames_are_returned_untouched() -> None:
    landmarker = FakeLandmarker()
    pipeline = _pipeline(landmarker)

    grayscale = np.zeros((20, 20), dtype=np.uint8)
    result = pipeline.process(grayscale)

    assert result.shape == (20, 20)
    assert pipeline.snapshot().status is TrackingStatus.ERROR
