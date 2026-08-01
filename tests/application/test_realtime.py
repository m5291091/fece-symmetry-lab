from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    FrameOrderError,
    LandmarkDetectionError,
)
from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode, FrameAdjustmentError
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)
from face_symmetry_lab.domain.models import AnalysisState, DemographicEstimate, Status

#: Spread-out points so `extract_face_crop` can derive a non-degenerate bbox.
SPREAD_POINTS = ((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5))


class FakeLandmarker:
    def __init__(
        self,
        face_count: int = 1,
        points: tuple[tuple[float, float], ...] | None = None,
    ) -> None:
        self.face_count = face_count
        self.points = points
        self.error: Exception | None = None
        self.calls: list[tuple[int, int]] = []
        self.received_images: list[NDArray[np.uint8]] = []
        self.closed = False
        self.policy_updates: list[object] = []

    def update_policy(self, policy: object) -> None:
        self.policy_updates.append(policy)

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.calls.append((frame_id, timestamp_ms))
        self.received_images.append(rgb_image)
        if self.error is not None:
            raise self.error
        points = (self.points or ((0.5, 0.5),)) if self.face_count == 1 else ()
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=self.face_count,
            points=points,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
        )

    def close(self) -> None:
        self.closed = True


class FakeSegmenter:
    def __init__(self, mask_value: float = 1.0) -> None:
        self.mask_value = mask_value
        self.error: Exception | None = None
        self.calls: list[int] = []
        self.received_images: list[NDArray[np.uint8]] = []
        self.closed = False

    def segment(self, rgb_image: NDArray[np.uint8], *, timestamp_ms: int) -> NDArray[np.float32]:
        self.calls.append(timestamp_ms)
        self.received_images.append(rgb_image)
        if self.error is not None:
            raise self.error
        height, width = rgb_image.shape[:2]
        return np.full((height, width), self.mask_value, dtype=np.float32)

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
        self.update_calls: list[tuple[object, float]] = []

    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: np.ndarray | None = None,
    ) -> AnalysisOutcome:
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

    def update(self, *, thresholds: object, tau_seconds: float) -> None:
        self.update_calls.append((thresholds, tau_seconds))


class RecorderSpy:
    def __init__(self) -> None:
        self.analysis_frames: list[int] = []
        self.error_frames: list[int] = []
        self.reset_calls = 0
        self.reset_kwargs: list[dict[str, object]] = []
        self.demographic_attachments: list[tuple[DemographicEstimate, int]] = []

    def append_analysis(self, frame: LandmarkFrame, outcome: AnalysisOutcome, **kwargs) -> None:
        self.analysis_frames.append(frame.frame_id)

    def append_error(self, *, frame_id: int, **kwargs) -> None:
        self.error_frames.append(frame_id)

    def reset(self, **kwargs) -> None:
        self.reset_calls += 1
        self.reset_kwargs.append(kwargs)

    def export_bundle(self) -> str:
        return "bundle"

    def attach_demographic_estimate(
        self,
        estimate: DemographicEstimate,
        *,
        observed_elapsed_ms: int,
    ) -> None:
        self.demographic_attachments.append((estimate, observed_elapsed_ms))


class FakeDemographicWorker:
    def __init__(self) -> None:
        self.submitted: list[NDArray[np.uint8]] = []
        self.reset_calls = 0
        self.closed = False
        self.due = True
        self._latest: tuple[DemographicEstimate, int] | None = None

    def is_due(self) -> bool:
        return self.due

    def submit(self, rgb_crop: NDArray[np.uint8]) -> None:
        self.submitted.append(rgb_crop)

    def set_latest(self, estimate: DemographicEstimate, version: int) -> None:
        self._latest = (estimate, version)

    def latest(self) -> tuple[DemographicEstimate, int] | None:
        return self._latest

    def reset(self) -> None:
        self.reset_calls += 1
        self._latest = None

    def close(self) -> None:
        self.closed = True


def _demographic_estimate(age: float = 30.0) -> DemographicEstimate:
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(0.4, 0.6),
        model_version="demographic/test",
        model_sha256="a" * 64,
        observed_at=datetime.now(UTC),
    )


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
    assert skipped_frame.sum() > 0
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
    assert raw.sum() > 0  # Short-lived geometry is retained; numeric analysis is not.


def test_cached_geometry_expires_before_the_next_slow_sample() -> None:
    clock = ManualClock()
    landmarker = FakeLandmarker()
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        clock=clock,
        analysis_interval_ms=1000,
        geometry_ttl_ms=250,
    )

    first = pipeline.process(_rgb())
    clock.advance(0.2)
    retained = pipeline.process(_rgb())
    clock.advance(0.051)
    expired = pipeline.process(_rgb())

    assert first.sum() > 0
    assert retained.sum() > 0
    assert expired.sum() == 0
    assert len(landmarker.calls) == 1


def test_reset_clears_cached_geometry_immediately() -> None:
    clock = ManualClock()
    pipeline = RealtimeLandmarkPipeline(
        landmarker=FakeLandmarker(),
        clock=clock,
        analysis_interval_ms=1000,
    )
    pipeline.process(_rgb())

    pipeline.reset_analysis()
    raw = pipeline.process(_rgb())

    assert raw.sum() > 0  # reset admits a fresh sample, never a cached one


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


def test_realtime_face_midline_can_be_turned_off_without_restarting() -> None:
    points = ((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5))
    pipeline = _pipeline(
        FakeLandmarker(points=points),
        midline_indices=(0, 1, 2),
        point_color=(0, 0, 0),
        point_radius=0,
    )

    shown = pipeline.process(_rgb())
    pipeline.set_midline_enabled(False)
    hidden = pipeline.process(_rgb())

    assert shown.sum() > 0
    assert hidden.sum() == 0


def _flat_rgb(value: int, height: int = 20, width: int = 20) -> NDArray[np.uint8]:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_brightness_gain_reaches_both_the_detector_and_the_display() -> None:
    landmarker = FakeLandmarker(face_count=0)  # no overlay drawn, so display == input
    pipeline = _pipeline(landmarker, brightness_gain=2.0)

    painted = pipeline.process(_flat_rgb(100))

    assert np.all(landmarker.received_images[0] == 200)
    assert np.all(painted == 200)


def test_a_default_gain_of_one_leaves_frames_unchanged() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker)

    painted = pipeline.process(_flat_rgb(100))

    assert np.all(painted == 100)


def test_an_out_of_range_brightness_gain_is_refused_at_construction() -> None:
    with pytest.raises(FrameAdjustmentError):
        _pipeline(FakeLandmarker(), brightness_gain=10.0)


def test_set_brightness_gain_updates_the_running_pipeline() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker)

    pipeline.set_brightness_gain(0.5)
    painted = pipeline.process(_flat_rgb(100))

    assert np.all(painted == 50)


def test_background_mode_off_never_calls_the_segmenter() -> None:
    segmenter = FakeSegmenter()
    pipeline = _pipeline(FakeLandmarker(face_count=0), segmenter=segmenter)

    pipeline.process(_rgb())

    assert segmenter.calls == []


def test_background_replacement_composites_only_the_display_frame() -> None:
    landmarker = FakeLandmarker(face_count=0)
    segmenter = FakeSegmenter(mask_value=0.0)  # fully background
    pipeline = _pipeline(
        landmarker,
        segmenter=segmenter,
        background_mode=BackgroundMode.WHITE,
    )

    painted = pipeline.process(_flat_rgb(10))

    assert np.all(landmarker.received_images[0] == 10)  # detector input untouched
    assert np.all(painted == 255)  # display replaced with white


def test_background_replacement_fails_open_on_segmenter_error() -> None:
    landmarker = FakeLandmarker(face_count=0)
    segmenter = FakeSegmenter()
    segmenter.error = RuntimeError("segmentation graph failure")
    pipeline = _pipeline(
        landmarker,
        segmenter=segmenter,
        background_mode=BackgroundMode.BLACK,
    )

    painted = pipeline.process(_flat_rgb(77))

    assert np.all(painted == 77)  # fell back to the un-composited frame
    assert pipeline.snapshot().consecutive_errors == 0  # not counted as a stream failure


def test_set_background_mode_updates_the_running_pipeline() -> None:
    landmarker = FakeLandmarker(face_count=0)
    segmenter = FakeSegmenter(mask_value=0.0)
    pipeline = _pipeline(landmarker, segmenter=segmenter)

    before = pipeline.process(_flat_rgb(10))
    pipeline.set_background_mode(BackgroundMode.BLACK)
    after = pipeline.process(_flat_rgb(10))

    assert np.all(before == 10)  # OFF by default: untouched
    assert np.all(after == 0)  # BLACK: replaced


def test_apply_advanced_settings_updates_analyzer_landmarker_and_recorder() -> None:
    """§15.2: an Advanced settings apply must reach every affected component."""
    landmarker = FakeLandmarker()
    analyzer = StubAnalyzer()
    recorder = RecorderSpy()
    pipeline = _pipeline(landmarker, analyzer=analyzer, recorder=recorder)
    thresholds = object()
    zoom_policy = object()

    pipeline.apply_advanced_settings(
        thresholds=thresholds,
        tau_seconds=0.7,
        zoom_policy=zoom_policy,
    )

    assert analyzer.update_calls == [(thresholds, 0.7)]
    assert landmarker.policy_updates == [zoom_policy]
    assert recorder.reset_kwargs[-1] == {"thresholds": thresholds, "tau_seconds": 0.7}


def test_apply_advanced_settings_clears_cached_geometry_and_background() -> None:
    """A throttled frame right after applying settings must not paint stale

    cached geometry/background left over from before the change.
    """
    landmarker = FakeLandmarker(points=((0.5, 0.2), (0.5, 0.5), (0.5, 0.8)))
    segmenter = FakeSegmenter(mask_value=0.0)  # fully background -> composites solid white
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        analyzer=StubAnalyzer(),
        segmenter=segmenter,
        background_mode=BackgroundMode.WHITE,
        clock=lambda: 0.0,
        analysis_interval_ms=1000,
    )
    pipeline.process(_rgb())  # populates cached geometry + mask; result is white

    pipeline.apply_advanced_settings(thresholds=object(), tau_seconds=0.4, zoom_policy=object())
    # clock never advances, so this next call is throttled and must fall
    # back to the (now-cleared) cache instead of the stale white composite.
    reused = pipeline.process(_rgb())

    assert np.all(reused == 0)


def test_cached_frames_reuse_the_last_computed_mask_without_resegmenting() -> None:
    clock = ManualClock()
    landmarker = FakeLandmarker(face_count=0)
    segmenter = FakeSegmenter(mask_value=0.0)
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        segmenter=segmenter,
        background_mode=BackgroundMode.WHITE,
        clock=clock,
        analysis_interval_ms=1000,
    )

    analysed = pipeline.process(_flat_rgb(10))
    clock.advance(0.05)  # well inside the 125ms/1000ms cadence: this frame is throttled
    throttled = pipeline.process(_flat_rgb(10))

    assert np.all(analysed == 255)
    assert np.all(throttled == 255)  # still composited, from the cached mask
    assert segmenter.calls == [0]  # segmentation only ran once


def test_single_face_frames_submit_a_crop_to_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.process(_rgb(height=40, width=40))

    assert len(worker.submitted) == 1
    assert worker.submitted[0].shape[2] == 3


def test_no_crop_is_built_while_the_worker_is_not_due() -> None:
    """The worker's cadence is slower than the analysis cadence, so the

    expensive crop must be skipped rather than built and discarded.
    """
    worker = FakeDemographicWorker()
    worker.due = False
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.process(_rgb(height=40, width=40))

    assert worker.submitted == []


def test_multi_face_frames_do_not_submit_to_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(face_count=2), demographic_worker=worker)

    pipeline.process(_rgb())

    assert worker.submitted == []


def test_no_face_frames_do_not_submit_to_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(face_count=0), demographic_worker=worker)

    pipeline.process(_rgb())

    assert worker.submitted == []


def test_degenerate_landmarks_do_not_crash_or_submit_to_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(points=((0.5, 0.5),)), demographic_worker=worker)

    annotated = pipeline.process(_rgb())

    assert annotated.shape == (20, 20, 3)
    assert worker.submitted == []


def test_a_fresh_demographic_estimate_is_attached_to_the_snapshot() -> None:
    worker = FakeDemographicWorker()
    estimate = _demographic_estimate()
    worker.set_latest(estimate, version=1)
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.process(_rgb(height=40, width=40))

    published = pipeline.snapshot().demographic_estimate
    assert published is not None
    assert published.estimated_age_years == estimate.estimated_age_years
    assert published.is_cached is False


def test_a_reused_demographic_estimate_is_marked_as_cached() -> None:
    worker = FakeDemographicWorker()
    worker.set_latest(_demographic_estimate(), version=1)
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.process(_rgb(height=40, width=40))
    first = pipeline.snapshot().demographic_estimate
    pipeline.process(_rgb(height=40, width=40))
    second = pipeline.snapshot().demographic_estimate

    assert first is not None and first.is_cached is False
    assert second is not None and second.is_cached is True


def test_a_new_demographic_version_is_not_marked_cached() -> None:
    worker = FakeDemographicWorker()
    worker.set_latest(_demographic_estimate(age=30.0), version=1)
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)
    pipeline.process(_rgb(height=40, width=40))

    worker.set_latest(_demographic_estimate(age=45.0), version=2)
    pipeline.process(_rgb(height=40, width=40))

    latest = pipeline.snapshot().demographic_estimate
    assert latest is not None
    assert latest.estimated_age_years == 45.0
    assert latest.is_cached is False


def test_demographic_estimate_is_attached_to_the_session_recorder() -> None:
    worker = FakeDemographicWorker()
    worker.set_latest(_demographic_estimate(), version=1)
    recorder = RecorderSpy()
    pipeline = _pipeline(
        FakeLandmarker(points=SPREAD_POINTS),
        analyzer=StubAnalyzer(),
        recorder=recorder,
        demographic_worker=worker,
    )

    pipeline.process(_rgb(height=40, width=40))

    assert len(recorder.demographic_attachments) == 1
    attached_estimate, observed_elapsed_ms = recorder.demographic_attachments[0]
    assert attached_estimate.estimated_age_years == 30.0
    assert observed_elapsed_ms >= 0


def test_no_attachment_without_a_demographic_worker() -> None:
    recorder = RecorderSpy()
    pipeline = _pipeline(
        FakeLandmarker(points=SPREAD_POINTS),
        analyzer=StubAnalyzer(),
        recorder=recorder,
    )

    pipeline.process(_rgb(height=40, width=40))

    assert recorder.demographic_attachments == []


def test_reset_analysis_resets_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(), demographic_worker=worker)

    pipeline.reset_analysis()

    assert worker.reset_calls == 1


def test_apply_advanced_settings_resets_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(
        FakeLandmarker(),
        analyzer=StubAnalyzer(),
        recorder=RecorderSpy(),
        demographic_worker=worker,
    )

    pipeline.apply_advanced_settings(thresholds=object(), tau_seconds=0.4, zoom_policy=object())

    assert worker.reset_calls == 1


def test_close_closes_the_demographic_worker() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(), demographic_worker=worker)

    pipeline.close()

    assert worker.closed is True


def test_disabling_demographics_stops_submission_and_display() -> None:
    worker = FakeDemographicWorker()
    worker.set_latest(_demographic_estimate(), version=1)
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.set_demographic_enabled(False)
    pipeline.process(_rgb(height=40, width=40))

    assert worker.submitted == []
    assert pipeline.snapshot().demographic_estimate is None


def test_re_enabling_demographics_resumes_submission() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(FakeLandmarker(points=SPREAD_POINTS), demographic_worker=worker)

    pipeline.set_demographic_enabled(False)
    pipeline.process(_rgb(height=40, width=40))
    pipeline.set_demographic_enabled(True)
    pipeline.process(_rgb(height=40, width=40))

    assert len(worker.submitted) == 1
