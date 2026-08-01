"""Streamlit + streamlit-webrtc composition for realtime landmark preview.

Composition only: this module wires the MediaPipe adapter to the realtime use
case and renders the result. No geometry, no inference, and no model download
happens here. The WebRTC callback runs on a worker thread and therefore never
touches Streamlit state (§14.3).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeFaceLandmarker,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.realtime import (
    PipelineSnapshot,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.ui.model_setup import landmark_asset, resolve_ready_landmarker
from face_symmetry_lab.ui.scores import render_scores

#: §10.2 forbids STUN and TURN. An empty ICE list keeps the session on loopback.
LOOPBACK_RTC_CONFIGURATION: Final[dict[str, list[object]]] = {"iceServers": []}

#: FR-005: video only. No audio permission is requested.
MEDIA_STREAM_CONSTRAINTS: Final[dict[str, object]] = {
    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
    "audio": False,
}

WEBRTC_KEY = "face-landmark-preview"
PIPELINE_STATE_KEY = "landmark_pipeline"

#: §11 NFR-003: the result panel refreshes at 4-5 Hz, independent of video fps.
STATUS_REFRESH_INTERVAL = "200ms"

_STATUS_MESSAGES: Final[dict[TrackingStatus, str]] = {
    TrackingStatus.NO_FACE: "顔が見つかりません。カメラの正面に入ってください。",
    TrackingStatus.SINGLE_FACE: "1 名の顔を追跡しています。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけが写るようにしてください。",
    TrackingStatus.ERROR: "このフレームを処理できませんでした。そのままお待ちください。",
}

VideoFrameCallback = Callable[[av.VideoFrame], av.VideoFrame]


def build_video_frame_callback(pipeline: RealtimeLandmarkPipeline) -> VideoFrameCallback:
    """Adapt the realtime pipeline to the streamlit-webrtc frame callback."""

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            painted = pipeline.process(rgb)
            annotated = av.VideoFrame.from_ndarray(painted, format="rgb24")
        except Exception:
            # Returning the source frame keeps the preview alive (NFR-017).
            return frame
        if frame.pts is not None:
            annotated.pts = frame.pts
        if frame.time_base is not None:
            annotated.time_base = frame.time_base
        return annotated

    return callback


def _build_pipeline(model_path: Path) -> RealtimeLandmarkPipeline:
    landmarker = MediaPipeFaceLandmarker(
        model_path=model_path,
        model_version=landmark_asset().version,
    )
    return RealtimeLandmarkPipeline(
        landmarker=landmarker,
        analyzer=FrameAnalyzer(config=load_metric_config()),
    )


def _session_pipeline(model_path: Path) -> RealtimeLandmarkPipeline:
    """Keep one landmark runtime per Streamlit session (§13.4)."""
    pipeline = st.session_state.get(PIPELINE_STATE_KEY)
    if not isinstance(pipeline, RealtimeLandmarkPipeline):
        pipeline = _build_pipeline(model_path)
        st.session_state[PIPELINE_STATE_KEY] = pipeline
    return pipeline


def release_session_pipeline() -> None:
    """Close the landmark runtime and drop the session reference (FR-004, §10.4)."""
    pipeline = st.session_state.pop(PIPELINE_STATE_KEY, None)
    if isinstance(pipeline, RealtimeLandmarkPipeline):
        pipeline.close()


@st.fragment(run_every=STATUS_REFRESH_INTERVAL)
def _render_status_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """Poll the immutable snapshot at 5 Hz without blocking the main script."""
    _render_snapshot(pipeline.snapshot())


def _render_snapshot(snapshot: PipelineSnapshot) -> None:
    st.write(_STATUS_MESSAGES[snapshot.status])
    tracking, points, dropped = st.columns(3)
    tracking.metric("追跡状態", snapshot.status.value)
    points.metric("ランドマーク点数", snapshot.landmark_count)
    dropped.metric("スキップしたフレーム", snapshot.dropped_frames)
    st.caption(f"最新フレーム #{snapshot.frame_id} / 処理時間 {snapshot.processing_ms:.1f} ms")
    if snapshot.needs_recovery:
        st.warning("処理が続けて失敗しています。カメラを停止して再開してください。")

    if snapshot.analysis is not None:
        st.divider()
        render_scores(snapshot.analysis)


def render_camera_section() -> None:
    """Render the realtime landmark preview. Call only after consent."""
    st.subheader("リアルタイム ランドマーク表示")
    st.caption("映像は端末内のメモリだけで処理され、保存も外部送信もしません。")

    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return

    context = webrtc_streamer(
        key=WEBRTC_KEY,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=LOOPBACK_RTC_CONFIGURATION,
        media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
        video_frame_callback=build_video_frame_callback(pipeline),
        async_processing=True,
    )

    if not context.state.playing:
        st.info("`START` を押すと、ブラウザがカメラの許可を求めます。")
        return

    _render_status_panel(pipeline)

    st.caption("カメラ映像は `STOP` で停止します。")
    controls = st.columns(2)
    if controls[0].button("スコアと平滑化をリセット", key="reset_analysis"):
        pipeline.reset_analysis()
        st.rerun()
    if controls[1].button("推論を停止してモデルを解放", key="release_landmarker"):
        release_session_pipeline()
        st.rerun()
