from fractions import Fraction

import av
import numpy as np

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.ui.camera import (
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    build_video_frame_callback,
)


class StubLandmarker:
    """Reports one face, so geometry gets drawn (FR-013)."""

    def __init__(self, face_count: int = 1) -> None:
        self.sizes: list[tuple[int, int]] = []
        self.face_count = face_count

    def detect_video(
        self,
        rgb_image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.sizes.append((rgb_image.shape[1], rgb_image.shape[0]))
        points = ((0.5, 0.5),) if self.face_count == 1 else ()
        return build_landmark_frame(frame_id=frame_id, face_count=self.face_count, points=points)

    def close(self) -> None:
        return None


class StubAnalyzer:
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


class BlackoutSegmenter:
    """Reports everything as background, so compositing visibly changes pixels."""

    def segment(self, rgb_image: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        height, width = rgb_image.shape[:2]
        return np.zeros((height, width), dtype=np.float32)

    def close(self) -> None:
        return None


def _source_frame(height: int = 48, width: int = 64, value: int = 200) -> av.VideoFrame:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
    frame.pts = 1
    frame.time_base = Fraction(1, 30)
    return frame


def test_webrtc_configuration_uses_no_external_ice_servers() -> None:
    assert LOOPBACK_RTC_CONFIGURATION == {"iceServers": []}


def test_media_constraints_request_video_only() -> None:
    assert MEDIA_STREAM_CONSTRAINTS["audio"] is False
    assert MEDIA_STREAM_CONSTRAINTS["video"] is not False


def test_callback_draws_geometry_onto_the_returned_frame() -> None:
    """FR-013: landmarks and the midline are the only things drawn."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analyzer=StubAnalyzer())  # type: ignore[arg-type]
    callback = build_video_frame_callback(pipeline)
    source = _source_frame(value=0)

    result = callback(source)

    assert result is not source
    assert result.to_ndarray(format="rgb24").sum() > 0


def test_callback_returns_the_source_frame_when_there_is_nothing_to_draw() -> None:
    """With no face detected there is no geometry, so the frame that arrived
    is handed straight back — no pixel change and no re-encode."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(face_count=0),  # type: ignore[arg-type]
        analyzer=StubAnalyzer(),  # type: ignore[arg-type]
    )
    callback = build_video_frame_callback(pipeline)
    source = _source_frame()

    result = callback(source)

    assert result is source


def test_callback_still_feeds_every_frame_to_the_pipeline() -> None:
    """Pass-through display must not mean pass-through analysis."""
    landmarker = StubLandmarker()
    callback = build_video_frame_callback(RealtimeLandmarkPipeline(landmarker=landmarker))

    callback(_source_frame())

    assert landmarker.sizes == [(64, 48)]


def test_analysis_still_reaches_the_snapshot_the_side_panel_reads() -> None:
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analyzer=StubAnalyzer())  # type: ignore[arg-type]
    callback = build_video_frame_callback(pipeline)

    callback(_source_frame())

    snapshot = pipeline.snapshot()
    assert snapshot.analysis is not None
    assert snapshot.frame_id == 1


def test_background_replacement_still_produces_a_new_frame_with_timing_intact() -> None:
    """The one case that legitimately rewrites pixels keeps pts/time_base."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(face_count=0),  # type: ignore[arg-type]
        segmenter=BlackoutSegmenter(),  # type: ignore[arg-type]
        background_mode=BackgroundMode.BLACK,
    )
    callback = build_video_frame_callback(pipeline)
    source = _source_frame(value=200)

    result = callback(source)

    assert result is not source
    assert (result.width, result.height) == (64, 48)
    assert result.pts == source.pts
    assert result.time_base == source.time_base
    assert result.to_ndarray(format="rgb24").max() == 0  # fully replaced


def test_callback_never_raises_when_the_pipeline_fails() -> None:
    class BrokenPipeline:
        def process_with_analysis(self, rgb_image: np.ndarray) -> tuple[np.ndarray, None]:
            raise RuntimeError("pipeline failure")

    callback = build_video_frame_callback(BrokenPipeline())  # type: ignore[arg-type]

    result = callback(_source_frame(height=8, width=8))

    assert isinstance(result, av.VideoFrame)


def test_worker_thread_callback_never_touches_streamlit_state() -> None:
    """§14.3: the WebRTC callback runs off-thread and must not call `st.*`."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera.build_video_frame_callback)

    assert "session_state" not in source
    assert "st." not in source
