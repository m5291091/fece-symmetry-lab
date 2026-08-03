from fractions import Fraction

import av
import numpy as np
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.ui.camera import (
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    SYMMETRIZATION_STATE_KEY,
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
            overall_distance_smooth=None,
            pose=None,
            face_area_ratio=None,
            metric_version="metric/test",
            landmark_map_version="map/test",
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


# --- FR-034 capture wiring ---------------------------------------------------


#: A symmetric face, spread out enough to fold about a real midline.
_SYMMETRIZABLE_POINTS = (
    (0.35, 0.40),
    (0.45, 0.40),
    (0.55, 0.40),
    (0.65, 0.40),
    (0.50, 0.30),
    (0.50, 0.50),
    (0.50, 0.62),
    (0.30, 0.55),
    (0.70, 0.55),
    (0.32, 0.46),
    (0.68, 0.46),
    (0.46, 0.52),
    (0.54, 0.52),
    (0.44, 0.58),
    (0.56, 0.58),
)


class SymmetrizableLandmarker(StubLandmarker):
    """Reports a face whose landmarks the versioned map can actually fold."""

    def detect_video(
        self,
        rgb_image: np.ndarray,
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        self.sizes.append((rgb_image.shape[1], rgb_image.shape[0]))
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=1,
            points=_SYMMETRIZABLE_POINTS,
            width=rgb_image.shape[1],
            height=rgb_image.shape[0],
        )


def _symmetrizable_config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(_SYMMETRIZABLE_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10)), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((11, 12),), min_pairs=1),
            "mouth": PartMap(pairs=((13, 14),), min_pairs=1),
        },
    )


def _panel_script() -> str:
    return (
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.camera import _render_symmetrization_panel\n"
        "_render_symmetrization_panel(st.session_state['pipeline'])\n"
    )


def test_the_capture_button_arms_the_pipeline_without_stopping_it() -> None:
    """FR-034: requesting a capture must not disturb the running analysis."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analysis_interval_ms=0)
    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()
    assert not at.exception

    at.button(key="capture_symmetrization").click().run()

    assert pipeline.capture_pending is True


def test_a_captured_frame_becomes_a_symmetrization_view() -> None:
    """§14.3g: the held frame is consumed once and rendered as a still result."""
    pipeline = RealtimeLandmarkPipeline(
        landmarker=SymmetrizableLandmarker(),
        analyzer=FrameAnalyzer(config=_symmetrizable_config()),
        analysis_interval_ms=0,
    )
    pipeline.request_capture()
    pipeline.process(np.full((80, 80, 3), 30, dtype=np.uint8))

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    view = at.session_state[SYMMETRIZATION_STATE_KEY]
    assert not isinstance(view, str)
    assert view.symmetrized_point_count > 0
    assert pipeline.take_capture() is None  # consumed exactly once


def test_a_capture_that_cannot_be_symmetrized_is_reported_not_retried() -> None:
    pipeline = RealtimeLandmarkPipeline(
        landmarker=StubLandmarker(),  # a single degenerate point cannot fold
        analyzer=FrameAnalyzer(config=_symmetrizable_config()),
        analysis_interval_ms=0,
    )
    pipeline.request_capture()
    pipeline.process(np.full((40, 40, 3), 30, dtype=np.uint8))

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()

    assert not at.exception
    assert isinstance(at.session_state[SYMMETRIZATION_STATE_KEY], str)
    assert at.warning  # surfaced to the user rather than silently dropped


def test_clearing_the_view_also_drops_the_held_frame() -> None:
    """§10.4: no face-derived image is kept after the user clears the panel."""
    pipeline = RealtimeLandmarkPipeline(landmarker=StubLandmarker(), analysis_interval_ms=0)
    pipeline.request_capture()

    at = AppTest.from_string(_panel_script())
    at.session_state["pipeline"] = pipeline
    at.run()
    at.button(key="clear_symmetrization").click().run()

    assert pipeline.capture_pending is False
    assert pipeline.take_capture() is None
    assert SYMMETRIZATION_STATE_KEY not in at.session_state


def test_the_symmetrization_panel_avoids_value_laden_wording() -> None:
    """§5.3/§10.7: a symmetrized face is a construction, not an ideal."""
    import inspect

    from face_symmetry_lab.ui import camera

    source = inspect.getsource(camera._render_symmetrization_panel)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source
