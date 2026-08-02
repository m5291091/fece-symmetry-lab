from datetime import UTC, datetime

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    FrameOrderError,
    LandmarkDetectionError,
)
from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.auto_tuning import recommend_age_interval_ms
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.frame_adjustments import (
    MAX_BRIGHTNESS_GAIN,
    BackgroundMode,
    FrameAdjustmentError,
)
from face_symmetry_lab.domain.landmarks import (
    LandmarkFrame,
    TrackingStatus,
    build_landmark_frame,
)
from face_symmetry_lab.domain.metrics import PART_NAMES, PartMap, build_metric_config
from face_symmetry_lab.domain.models import AnalysisState, DemographicEstimate, Status
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

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
        self.policy: object | None = None

    def update_policy(self, policy: object) -> None:
        self.policy_updates.append(policy)
        self.policy = policy

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
        self.update_calls: list[tuple[object, float, object]] = []
        self.config = object()

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

    def update(
        self,
        *,
        thresholds: object,
        tau_seconds: float,
        config: object = None,
    ) -> None:
        self.update_calls.append((thresholds, tau_seconds, config))
        if config is not None:
            self.config = config


def _metric_config():
    return build_metric_config(
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
        sensitivity=dict.fromkeys(PART_NAMES, 30.0),
        weights=dict.fromkeys(PART_NAMES, 0.25),
        sensitivity_frozen=True,
    )


class NoisyAnalyzer:
    """A real config/thresholds/tau contract with an injectable distance signal.

    Used only by the sensitivity auto-tune tests, which need `with_tunables`
    to actually validate the recommended sensitivity against a real bundle.
    """

    def __init__(self, *, config=None, distances: dict[str, float] | None = None) -> None:
        self.config = config if config is not None else _metric_config()
        self._thresholds = QualityThresholds()
        self._tau_seconds = 0.4
        self.distances = distances if distances is not None else dict.fromkeys(PART_NAMES, 0.1)
        self.update_calls: list[tuple[object, float, object]] = []
        self.reset_calls = 0

    @property
    def thresholds(self) -> QualityThresholds:
        return self._thresholds

    @property
    def tau_seconds(self) -> float:
        return self._tau_seconds

    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: np.ndarray | None = None,
    ) -> AnalysisOutcome:
        symmetry = SymmetryResult(
            scores=dict.fromkeys(PART_NAMES, 90.0),
            distances=dict(self.distances),
            overall=90.0,
            metric_version=self.config.metric_version,
            landmark_map_version=self.config.landmark_map_version,
        )
        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=AnalysisState(status=Status.OK),
            symmetry=symmetry,
            overall_smooth=90.0,
            pose=frame.pose,
            face_area_ratio=frame.face_area_ratio,
            metric_version=self.config.metric_version,
            landmark_map_version=self.config.landmark_map_version,
            sensitivity_frozen=self.config.sensitivity_frozen,
        )

    def reset(self) -> None:
        self.reset_calls += 1

    def update(
        self,
        *,
        thresholds: object,
        tau_seconds: float,
        config: object = None,
    ) -> None:
        self.update_calls.append((thresholds, tau_seconds, config))
        self._thresholds = thresholds  # type: ignore[assignment]
        self._tau_seconds = tau_seconds
        if config is not None:
            self.config = config


class RecorderSpy:
    def __init__(self) -> None:
        self.analysis_frames: list[int] = []
        self.error_frames: list[int] = []
        self.reset_calls = 0
        self.reset_kwargs: list[dict[str, object]] = []
        self.demographic_attachments: list[tuple[DemographicEstimate, int]] = []
        self.resize_calls: list[int] = []

    def append_analysis(self, frame: LandmarkFrame, outcome: AnalysisOutcome, **kwargs) -> None:
        self.analysis_frames.append(frame.frame_id)

    def append_error(self, *, frame_id: int, **kwargs) -> None:
        self.error_frames.append(frame_id)

    def reset(self, **kwargs) -> None:
        self.reset_calls += 1
        self.reset_kwargs.append(kwargs)

    def resize_history_capacity(self, capacity: int) -> None:
        self.resize_calls.append(capacity)

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
        self.interval_updates: list[int] = []
        self._latest: tuple[DemographicEstimate, int] | None = None

    def is_due(self) -> bool:
        return self.due

    def submit(self, rgb_crop: NDArray[np.uint8]) -> None:
        self.submitted.append(rgb_crop)

    def set_interval_ms(self, interval_ms: int) -> None:
        self.interval_updates.append(interval_ms)

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

    assert analyzer.update_calls == [(thresholds, 0.7, None)]
    assert landmarker.policy_updates == [zoom_policy]
    assert recorder.reset_kwargs[-1] == {
        "thresholds": thresholds,
        "tau_seconds": 0.7,
        "metric_config": None,
        "history_capacity": None,
        "analysis_interval_ms": None,
    }


def test_metric_config_is_none_without_an_analyzer() -> None:
    pipeline = _pipeline(FakeLandmarker())

    assert pipeline.metric_config is None


def test_metric_config_reflects_the_attached_analyzers_bundle() -> None:
    analyzer = StubAnalyzer()
    pipeline = _pipeline(FakeLandmarker(), analyzer=analyzer)

    assert pipeline.metric_config is analyzer.config


def test_apply_advanced_settings_can_swap_the_metric_config() -> None:
    """§15.2: an edited part_weights/sensitivity bundle must reach analyzer + recorder."""
    landmarker = FakeLandmarker()
    analyzer = StubAnalyzer()
    recorder = RecorderSpy()
    pipeline = _pipeline(landmarker, analyzer=analyzer, recorder=recorder)
    metric_config = object()

    pipeline.apply_advanced_settings(
        thresholds=object(),
        tau_seconds=0.4,
        zoom_policy=object(),
        metric_config=metric_config,
    )

    assert analyzer.update_calls[-1][2] is metric_config
    assert recorder.reset_kwargs[-1]["metric_config"] is metric_config


def test_apply_advanced_settings_can_resize_history_capacity() -> None:
    recorder = RecorderSpy()
    pipeline = _pipeline(FakeLandmarker(), analyzer=StubAnalyzer(), recorder=recorder)

    pipeline.apply_advanced_settings(
        thresholds=object(),
        tau_seconds=0.4,
        zoom_policy=object(),
        history_capacity=500,
    )

    assert recorder.reset_kwargs[-1]["history_capacity"] == 500


def test_apply_advanced_settings_updates_the_demographic_worker_interval() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(
        FakeLandmarker(),
        analyzer=StubAnalyzer(),
        recorder=RecorderSpy(),
        demographic_worker=worker,
    )

    pipeline.apply_advanced_settings(
        thresholds=object(),
        tau_seconds=0.4,
        zoom_policy=object(),
        age_interval_ms=2000,
    )

    assert worker.interval_updates == [2000]


def test_apply_advanced_settings_without_an_age_interval_leaves_the_worker_untouched() -> None:
    worker = FakeDemographicWorker()
    pipeline = _pipeline(
        FakeLandmarker(),
        analyzer=StubAnalyzer(),
        recorder=RecorderSpy(),
        demographic_worker=worker,
    )

    pipeline.apply_advanced_settings(thresholds=object(), tau_seconds=0.4, zoom_policy=object())

    assert worker.interval_updates == []


def test_apply_advanced_settings_can_change_the_analysis_cadence() -> None:
    """§15.2 `analysis_interval_ms`: independent of the WebRTC frame rate."""
    clock = ManualClock()
    landmarker = FakeLandmarker()
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker, clock=clock, analysis_interval_ms=1000
    )
    pipeline.process(_rgb())  # consumes the first admitted slot

    pipeline.apply_advanced_settings(
        thresholds=object(),
        tau_seconds=0.4,
        zoom_policy=object(),
        analysis_interval_ms=0,
    )
    pipeline.process(_rgb())  # would be throttled under the old 1000ms cadence

    assert len(landmarker.calls) == 2


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


def test_auto_tune_is_off_by_default_and_never_touches_brightness_gain() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker)

    pipeline.process(_flat_rgb(20))  # far from the 128 target

    assert pipeline.brightness_gain == pytest.approx(1.0)


def test_auto_tune_enabled_moves_brightness_gain_toward_the_target() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker, auto_tune_enabled=True)

    pipeline.process(_flat_rgb(64))  # 128 target / 64 measured -> gain 2.0

    assert pipeline.brightness_gain == pytest.approx(2.0)


def test_auto_tuned_brightness_is_clamped_to_fr_028_bounds() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker, auto_tune_enabled=True)

    pipeline.process(_flat_rgb(1))  # would recommend a gain far above 3.0

    assert pipeline.brightness_gain == pytest.approx(MAX_BRIGHTNESS_GAIN)


def test_set_auto_tune_enabled_can_be_turned_off_again() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker, auto_tune_enabled=True)

    pipeline.process(_flat_rgb(64))
    pipeline.set_auto_tune_enabled(False)
    pipeline.process(_flat_rgb(1))  # would move gain again if auto-tune were still on

    assert pipeline.brightness_gain == pytest.approx(2.0)


def test_auto_tune_enabled_updates_the_landmarker_zoom_policy_from_face_area() -> None:
    landmarker = FakeLandmarker(points=SPREAD_POINTS)
    pipeline = _pipeline(landmarker, auto_tune_enabled=True)

    pipeline.process(_rgb(height=40, width=40))

    assert landmarker.policy_updates != []
    assert pipeline.zoom_policy is landmarker.policy
    assert pipeline.zoom_policy.target_px != TrackingZoomPolicy().target_px


def test_auto_tune_disabled_never_calls_update_policy() -> None:
    landmarker = FakeLandmarker(points=SPREAD_POINTS)
    pipeline = _pipeline(landmarker)

    pipeline.process(_rgb(height=40, width=40))

    assert landmarker.policy_updates == []


def test_auto_tune_zoom_is_skipped_without_a_single_confirmed_face() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = _pipeline(landmarker, auto_tune_enabled=True)

    pipeline.process(_rgb())

    assert landmarker.policy_updates == []


def test_auto_tune_zoom_no_ops_when_the_landmarker_does_not_support_zoom() -> None:
    class PlainLandmarker:
        def detect_video(
            self, rgb_image: NDArray[np.uint8], *, frame_id: int, timestamp_ms: int
        ) -> LandmarkFrame:
            return build_landmark_frame(
                frame_id=frame_id,
                face_count=1,
                points=SPREAD_POINTS,
                width=rgb_image.shape[1],
                height=rgb_image.shape[0],
            )

        def close(self) -> None:
            pass

    pipeline = RealtimeLandmarkPipeline(
        landmarker=PlainLandmarker(),
        clock=StepClock(),
        analysis_interval_ms=0,
        auto_tune_enabled=True,
    )

    pipeline.process(_rgb(height=40, width=40))

    assert pipeline.zoom_policy is None
    assert pipeline.snapshot().status is TrackingStatus.SINGLE_FACE


def test_auto_tune_enabled_slows_the_analysis_interval_under_heavy_processing_load() -> None:
    landmarker = FakeLandmarker(face_count=0)
    recorder = RecorderSpy()
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        recorder=recorder,
        clock=StepClock(step=0.2),  # every clock() call is 200ms apart
        analysis_interval_ms=0,
        auto_tune_enabled=True,
    )

    for _ in range(10):
        pipeline.process(_rgb())

    # 200ms mean load only clears the slowest documented choice's utilization target.
    assert pipeline.analysis_interval_ms == 333
    assert recorder.resize_calls != []
    assert recorder.reset_calls == 0  # cadence retuning must not fragment history


def test_auto_tune_enabled_speeds_the_analysis_interval_back_up_under_light_load() -> None:
    landmarker = FakeLandmarker(face_count=0)
    pipeline = RealtimeLandmarkPipeline(
        landmarker=landmarker,
        clock=StepClock(step=0.2),
        analysis_interval_ms=333,
        auto_tune_enabled=True,
    )
    for _ in range(10):  # let it settle at the slow interval first
        pipeline.process(_rgb())
    assert pipeline.analysis_interval_ms == 333

    # A lighter but still admission-gate-realistic clock (50ms/tick, under the
    # 125-choice's ~87.5ms utilization threshold either way).
    fast_pipeline = RealtimeLandmarkPipeline(
        landmarker=FakeLandmarker(face_count=0),
        clock=StepClock(step=0.05),
        analysis_interval_ms=333,
        auto_tune_enabled=True,
    )
    for _ in range(100):
        fast_pipeline.process(_rgb())

    assert fast_pipeline.analysis_interval_ms == 125


def test_auto_tune_disabled_never_changes_the_analysis_interval() -> None:
    pipeline = RealtimeLandmarkPipeline(
        landmarker=FakeLandmarker(face_count=0),
        clock=StepClock(step=0.2),
        analysis_interval_ms=125,
    )

    for _ in range(10):
        pipeline.process(_rgb())

    assert pipeline.analysis_interval_ms == 125


def test_auto_tune_enabled_retunes_the_demographic_worker_interval_with_cadence() -> None:
    worker = FakeDemographicWorker()
    pipeline = RealtimeLandmarkPipeline(
        landmarker=FakeLandmarker(face_count=0),
        demographic_worker=worker,
        clock=StepClock(step=0.2),
        analysis_interval_ms=0,
        auto_tune_enabled=True,
    )

    for _ in range(10):
        pipeline.process(_rgb())

    assert worker.interval_updates != []
    assert worker.interval_updates[-1] == recommend_age_interval_ms(333)


def test_auto_tune_enabled_retunes_sensitivity_from_a_noisy_part() -> None:
    landmarker = FakeLandmarker(points=SPREAD_POINTS)
    analyzer = NoisyAnalyzer(distances=dict.fromkeys(PART_NAMES, 0.1))
    recorder = RecorderSpy()
    pipeline = _pipeline(landmarker, analyzer=analyzer, recorder=recorder, auto_tune_enabled=True)

    for i in range(80):
        analyzer.distances["contour"] = 0.05 if i % 2 == 0 else 0.35
        pipeline.process(_rgb(height=40, width=40))

    assert analyzer.update_calls != []
    assert analyzer.config.sensitivity["contour"] != pytest.approx(30.0)
    assert analyzer.config.sensitivity["eyes"] == pytest.approx(30.0)
    assert analyzer.config.weights == dict.fromkeys(PART_NAMES, 0.25)  # part_weights frozen
    assert analyzer.config.sensitivity_frozen is False
    assert recorder.reset_calls >= 1  # a real scoring change starts a new segment


def test_auto_tune_enabled_does_not_retune_sensitivity_when_distances_are_stable() -> None:
    landmarker = FakeLandmarker(points=SPREAD_POINTS)
    analyzer = NoisyAnalyzer(distances=dict.fromkeys(PART_NAMES, 0.1))
    recorder = RecorderSpy()
    pipeline = _pipeline(landmarker, analyzer=analyzer, recorder=recorder, auto_tune_enabled=True)

    for _ in range(80):
        pipeline.process(_rgb(height=40, width=40))

    assert analyzer.update_calls == []
    assert recorder.reset_calls == 0


def test_auto_tune_disabled_never_retunes_sensitivity() -> None:
    landmarker = FakeLandmarker(points=SPREAD_POINTS)
    analyzer = NoisyAnalyzer(distances=dict.fromkeys(PART_NAMES, 0.1))
    pipeline = _pipeline(landmarker, analyzer=analyzer)

    for i in range(80):
        analyzer.distances["contour"] = 0.05 if i % 2 == 0 else 0.35
        pipeline.process(_rgb(height=40, width=40))

    assert analyzer.update_calls == []
