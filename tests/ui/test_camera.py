from datetime import UTC, datetime
from fractions import Fraction

import av
import numpy as np

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.models import AnalysisState, DemographicEstimate, Status
from face_symmetry_lab.ui.camera import (
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    build_video_frame_callback,
)
from face_symmetry_lab.ui.camera_overlay import OverlayPosition


class FakeDemographicWorker:
    """Minimal `DemographicSampler` double: always has one estimate ready."""

    def __init__(self, estimate: DemographicEstimate) -> None:
        self._estimate = estimate

    def is_due(self) -> bool:
        return False

    def submit(self, rgb_crop: np.ndarray) -> None:
        return None

    def latest(self) -> tuple[DemographicEstimate, int]:
        return self._estimate, 1

    def reset(self) -> None:
        return None

    def set_interval_ms(self, interval_ms: int) -> None:
        return None

    def close(self) -> None:
        return None


class StubLandmarker:
    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def detect_video(
        self,
        rgb_image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.sizes.append((rgb_image.shape[1], rgb_image.shape[0]))
        return build_landmark_frame(frame_id=frame_id, face_count=1, points=((0.5, 0.5),))

    def close(self) -> None:
        return None


class OverlayAnalyzer:
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
            pose=None,
            face_area_ratio=None,
            metric_version="metric/test",
            landmark_map_version="map/test",
            sensitivity_frozen=False,
        )

    def reset(self) -> None:
        return None


def test_webrtc_configuration_uses_no_external_ice_servers() -> None:
    assert LOOPBACK_RTC_CONFIGURATION == {"iceServers": []}


def test_media_constraints_request_video_only() -> None:
    assert MEDIA_STREAM_CONSTRAINTS["audio"] is False
    assert MEDIA_STREAM_CONSTRAINTS["video"] is not False


def test_callback_returns_a_video_frame_of_the_same_size() -> None:
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker())
    callback = build_video_frame_callback(pipeline)
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
    source.pts = 1
    source.time_base = Fraction(1, 30)

    result = callback(source)

    assert isinstance(result, av.VideoFrame)
    assert (result.width, result.height) == (64, 48)
    assert result.pts == source.pts
    assert result.time_base == source.time_base


def test_callback_feeds_rgb_pixels_to_the_pipeline() -> None:
    landmarker = StubLandmarker()
    callback = build_video_frame_callback(RealtimeLandmarkPipeline(landmarker=landmarker))
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")

    callback(source)

    assert landmarker.sizes == [(64, 48)]


def test_callback_draws_the_overlay_into_the_returned_frame() -> None:
    callback = build_video_frame_callback(RealtimeLandmarkPipeline(landmarker=StubLandmarker()))
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")

    result = callback(source).to_ndarray(format="rgb24")

    assert result.sum() > 0


def test_callback_draws_numeric_status_over_the_camera_image() -> None:
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),
        analyzer=OverlayAnalyzer(),  # type: ignore[arg-type]
    )
    callback = build_video_frame_callback(pipeline)
    source = av.VideoFrame.from_ndarray(np.zeros((720, 1280, 3), dtype=np.uint8), format="rgb24")

    result = callback(source).to_ndarray(format="rgb24")

    assert result[:150, :500].sum() > 0


def test_callback_keeps_latest_numbers_and_short_lived_geometry(monkeypatch) -> None:
    from face_symmetry_lab.ui import camera

    overlay_calls: list[tuple[AnalysisOutcome | None, bool]] = []

    def record_overlay(
        rgb_image: np.ndarray,
        outcome: AnalysisOutcome | None,
        *,
        current_sample: bool,
        demographic: object | None = None,
        position: object | None = None,
    ) -> np.ndarray:
        overlay_calls.append((outcome, current_sample))
        return rgb_image

    monkeypatch.setattr(camera, "draw_camera_metrics", record_overlay)
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),
        analyzer=OverlayAnalyzer(),  # type: ignore[arg-type]
        clock=lambda: 0.0,
    )
    callback = build_video_frame_callback(pipeline)
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")

    first = callback(source).to_ndarray(format="rgb24")
    second = callback(source).to_ndarray(format="rgb24")

    assert overlay_calls[0][0] is not None
    assert overlay_calls[0][1] is True
    assert overlay_calls[1][0] is overlay_calls[0][0]
    assert overlay_calls[1][1] is False
    assert first.sum() > 0
    assert second.sum() > 0  # Geometry is retained briefly to avoid visible blinking.


def test_callback_forwards_the_configured_position_to_the_overlay(monkeypatch) -> None:
    from face_symmetry_lab.ui import camera

    calls: list[object] = []

    def record_overlay(rgb_image, outcome, *, current_sample, demographic=None, position=None):
        calls.append(position)
        return rgb_image

    monkeypatch.setattr(camera, "draw_camera_metrics", record_overlay)
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker())
    callback = build_video_frame_callback(pipeline, position=OverlayPosition.BOTTOM_RIGHT)
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")

    callback(source)

    assert calls == [OverlayPosition.BOTTOM_RIGHT]


def test_callback_defaults_to_the_top_left_position() -> None:
    """Backward compatible default: unset position behaves as it always did."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),
        analyzer=OverlayAnalyzer(),  # type: ignore[arg-type]
    )
    callback = build_video_frame_callback(pipeline)
    source = av.VideoFrame.from_ndarray(np.zeros((720, 1280, 3), dtype=np.uint8), format="rgb24")

    result = callback(source).to_ndarray(format="rgb24")

    assert result[:150, :500].sum() > 0  # same region the existing top-left test checks


def test_callback_forwards_the_snapshot_demographic_estimate_to_the_overlay(monkeypatch) -> None:
    from face_symmetry_lab.ui import camera

    estimate = build_demographic_estimate(
        estimated_age_years=30.0,
        gender_probabilities=(0.4, 0.6),
        model_version="demographic/test",
        model_sha256="a" * 64,
        observed_at=datetime.now(UTC),
    )
    calls: list[object] = []

    def record_overlay(rgb_image, outcome, *, current_sample, demographic=None, position=None):
        calls.append(demographic)
        return rgb_image

    monkeypatch.setattr(camera, "draw_camera_metrics", record_overlay)
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),
        demographic_worker=FakeDemographicWorker(estimate),
    )
    callback = build_video_frame_callback(pipeline)
    source = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")

    callback(source)

    assert calls == [estimate]


def test_callback_never_raises_when_the_pipeline_fails() -> None:
    class BrokenPipeline:
        def process(self, rgb_image: np.ndarray) -> np.ndarray:
            raise RuntimeError("pipeline failure")

    callback = build_video_frame_callback(BrokenPipeline())  # type: ignore[arg-type]
    source = av.VideoFrame.from_ndarray(np.zeros((8, 8, 3), dtype=np.uint8), format="rgb24")

    result = callback(source)

    assert isinstance(result, av.VideoFrame)


def test_worker_thread_callback_never_touches_streamlit_state() -> None:
    """§14.3: the WebRTC callback runs off-thread and must not call `st.*`."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "session_state" not in source
    assert "st." not in source
