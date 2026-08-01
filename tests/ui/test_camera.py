from fractions import Fraction

import av
import numpy as np

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.realtime import RealtimeLandmarkPipeline
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.ui.camera import (
    LOOPBACK_RTC_CONFIGURATION,
    MEDIA_STREAM_CONSTRAINTS,
    build_video_frame_callback,
)


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
    def analyze(self, frame: LandmarkFrame, *, at: float) -> AnalysisOutcome:
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
