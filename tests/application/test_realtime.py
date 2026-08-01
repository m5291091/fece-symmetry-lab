import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    FrameOrderError,
    LandmarkDetectionError,
)
from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)
from face_symmetry_lab.domain.models import AnalysisState, Status


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


class ManualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubAnalyzer:
    def __init__(self) -> None:
        self.reset_calls = 0

    def analyze(self, frame: LandmarkFrame, *, at: float) -> AnalysisOutcome:
        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=AnalysisState(status=Status.HOLD),
            symmetry=None,
            overall_smooth=None,
            pose=frame.pose,
            face_area_ratio=frame.face_area_ratio,
            metric_version="metric/test",
            landmark_map_version="map/test",
            sensitivity_frozen=False,
        )

    def reset(self) -> None:
        self.reset_calls += 1


class RecorderSpy:
    def __init__(self) -> None:
        self.analysis_frames: list[int] = []
        self.error_frames: list[int] = []
        self.reset_calls = 0

    def append_analysis(self, frame: LandmarkFrame, outcome: AnalysisOutcome, **kwargs) -> None:
        self.analysis_frames.append(frame.frame_id)

    def append_error(self, *, frame_id: int, **kwargs) -> None:
        self.error_frames.append(frame_id)

    def reset(self) -> None:
        self.reset_calls += 1

    def export_bundle(self) -> str:
        return "bundle"


def _rgb(height: int = 20, width: int = 20) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _pipeline(landmarker: FakeLandmarker, **kwargs: object) -> RealtimeLandmarkPipeline:
    return RealtimeLandmarkPipeline(
        landmarker=landmarker,
        clock=StepClock(),
        analysis_interval_ms=0,
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
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        clock=lambda: 1.0,
        analysis_interval_ms=0,
    )

    pipeline.process(_rgb())
    pipeline.process(_rgb())

    timestamps = [call[1] for call in landmarker.calls]
    assert timestamps[1] > timestamps[0]


def test_pipeline_samples_latest_frames_at_default_cadence() -> None:
    clock = ManualClock()
    landmarker = FakeLandmarker()
    pipeline = RealtimeLandmarkPipeline(landmarker=landmarker, clock=clock)

    _, first = pipeline.process_with_analysis(_rgb())
    skipped_frame, second = pipeline.process_with_analysis(_rgb())
    clock.advance(0.124)
    pipeline.process_with_analysis(_rgb())
    clock.advance(0.001)
    _, fourth = pipeline.process_with_analysis(_rgb())

    assert first is None  # No analyzer is attached, but landmark inference ran.
    assert second is None
    assert fourth is None
    assert skipped_frame.sum() == 0
    assert len(landmarker.calls) == 2
    assert pipeline.snapshot().throttled_frames == 2


def test_pipeline_reset_allows_an_immediate_sample() -> None:
    clock = ManualClock()
    landmarker = FakeLandmarker()
    pipeline = RealtimeLandmarkPipeline(landmarker=landmarker, clock=clock)

    pipeline.process(_rgb())
    pipeline.process(_rgb())
    pipeline.reset_analysis()
    pipeline.process(_rgb())

    assert len(landmarker.calls) == 2
    assert pipeline.snapshot().throttled_frames == 0


def test_a_busy_analyzer_returns_the_raw_frame_instead_of_queueing() -> None:
    landmarker = FakeLandmarker()
    pipeline = _pipeline(landmarker)

    with pipeline.scheduler.slot():
        annotated = pipeline.process(_rgb())

    assert annotated.sum() == 0
    assert landmarker.calls == []
    assert pipeline.snapshot().dropped_frames == 1


def test_a_dropped_frame_never_receives_the_previous_numeric_overlay_result() -> None:
    pipeline = _pipeline(FakeLandmarker(), analyzer=StubAnalyzer())
    _, first_analysis = pipeline.process_with_analysis(_rgb())

    with pipeline.scheduler.slot():
        raw, dropped_analysis = pipeline.process_with_analysis(_rgb())

    assert first_analysis is not None
    assert dropped_analysis is None
    assert raw.sum() == 0


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


def test_successful_analysis_is_added_to_the_session_export_history() -> None:
    recorder = RecorderSpy()
    pipeline = _pipeline(
        FakeLandmarker(),
        analyzer=StubAnalyzer(),
        recorder=recorder,
    )

    pipeline.process(_rgb())

    assert recorder.analysis_frames == [1]
    assert pipeline.export_bundle() == "bundle"


def test_detection_errors_are_added_without_stopping_future_export() -> None:
    landmarker = FakeLandmarker()
    landmarker.error = LandmarkDetectionError("boom")
    recorder = RecorderSpy()
    pipeline = _pipeline(landmarker, recorder=recorder)

    pipeline.process(_rgb())

    assert recorder.error_frames == [1]


def test_reset_clears_the_export_history_together_with_analysis_state() -> None:
    analyzer = StubAnalyzer()
    recorder = RecorderSpy()
    pipeline = _pipeline(FakeLandmarker(), analyzer=analyzer, recorder=recorder)

    pipeline.reset_analysis()

    assert analyzer.reset_calls == 1
    assert recorder.reset_calls == 1
