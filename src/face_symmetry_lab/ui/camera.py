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

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.export.session_exporter import (
    csv_bytes,
    export_filename,
    json_bytes,
)
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeFaceLandmarker,
)
from face_symmetry_lab.adapters.inference.mediapipe_segmenter import (
    MediaPipeSegmenter,
    SegmentationError,
)
from face_symmetry_lab.adapters.inference.tracking_zoom_landmarker import (
    TrackingZoomLandmarker,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import InstallStatus, file_sha256
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.quality import QualityConfigError, QualityThresholds
from face_symmetry_lab.application.realtime import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    PipelineSnapshot,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.frame_adjustments import (
    MAX_BRIGHTNESS_GAIN,
    MIN_BRIGHTNESS_GAIN,
    BackgroundMode,
)
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomError, TrackingZoomPolicy
from face_symmetry_lab.ui.camera_overlay import draw_camera_metrics
from face_symmetry_lab.ui.model_setup import (
    configured_segmenter_path,
    landmark_asset,
    resolve_ready_landmarker,
    segmenter_asset,
    segmenter_status,
)
from face_symmetry_lab.ui.scores import render_scores

_BACKGROUND_MODE_LABELS: Final[dict[BackgroundMode, str]] = {
    BackgroundMode.OFF: "しない",
    BackgroundMode.BLACK: "黒",
    BackgroundMode.WHITE: "白",
}

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
LOCK_PATH = Path(__file__).resolve().parents[3] / "uv.lock"

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
            painted, analysis = pipeline.process_with_analysis(rgb)
            latest = analysis if analysis is not None else pipeline.snapshot().analysis
            painted = draw_camera_metrics(
                painted,
                latest,
                current_sample=analysis is not None,
            )
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


def _build_segmenter() -> MediaPipeSegmenter | None:
    """Background replacement is optional (§14.3b): never blocks the camera."""
    path = configured_segmenter_path()
    if segmenter_status(path) is not InstallStatus.READY:
        return None
    try:
        return MediaPipeSegmenter(model_path=path, model_version=segmenter_asset().version)
    except SegmentationError:
        return None


def _build_pipeline(model_path: Path) -> RealtimeLandmarkPipeline:
    config = load_metric_config()
    landmarker = MediaPipeFaceLandmarker(
        model_path=model_path,
        model_version=landmark_asset().version,
    )
    # §14.3a: a PC webcam frames a face far smaller than a phone selfie, so
    # crop-and-enlarge around the last detection before handing it to
    # MediaPipe. The wrapper still reports `model_version`/`model_sha256`
    # via the underlying `landmarker` reference below.
    zoomed_landmarker = TrackingZoomLandmarker(landmarker)
    analyzer = FrameAnalyzer(config=config)
    recorder = SessionRecorder(
        metric_config=config,
        thresholds=analyzer.thresholds,
        tau_seconds=DEFAULT_TAU_SECONDS,
        landmark_model_version=landmarker.model_version,
        landmark_model_sha256=landmarker.model_sha256,
        app_version=__version__,
        lock_digest=file_sha256(LOCK_PATH),
        analysis_interval_ms=DEFAULT_ANALYSIS_INTERVAL_MS,
    )
    return RealtimeLandmarkPipeline(
        landmarker=zoomed_landmarker,
        analyzer=analyzer,
        recorder=recorder,
        segmenter=_build_segmenter(),
        analysis_interval_ms=DEFAULT_ANALYSIS_INTERVAL_MS,
        midline_indices=config.midline_indices,
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
    with st.container(border=True):
        message = _STATUS_MESSAGES[snapshot.status]
        if snapshot.status is TrackingStatus.ERROR:
            st.error(message)
        elif snapshot.status is TrackingStatus.MULTI_FACE:
            st.warning(message)
        elif snapshot.status is TrackingStatus.SINGLE_FACE:
            st.success(message)
        else:
            st.info(message)

        tracking, points = st.columns(2)
        tracking.metric("追跡状態", snapshot.status.value)
        points.metric("ランドマーク点数", snapshot.landmark_count)
        sampled, busy = st.columns(2)
        sampled.metric("プレビュー専用フレーム", snapshot.throttled_frames)
        busy.metric("処理中スキップ", snapshot.dropped_frames)
        st.caption(
            f"解析サンプル #{snapshot.frame_id} · 処理 {snapshot.processing_ms:.1f} ms · "
            f"最大 {1000 // DEFAULT_ANALYSIS_INTERVAL_MS} Hz"
        )
        if snapshot.needs_recovery:
            st.warning("処理が続けて失敗しています。カメラを停止して再開してください。")

        if snapshot.analysis is not None:
            st.divider()
            st.caption("カメラ映像の LAST SAMPLE と同じ直近の解析値です。")
            render_scores(snapshot.analysis)


def _render_export_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    bundle = pipeline.export_bundle()
    if bundle is None or not bundle.records:
        return

    st.divider()
    st.subheader("セッションデータのエクスポート")
    st.warning(
        "姿勢・左右差などの顔由来の派生データを含みます。"
        "共有先と利用目的を確認してから保存してください。"
    )
    revision = bundle.manifest.config_revisions[-1]
    st.caption(
        f"{len(bundle.records)} 件 · metric {bundle.manifest.metric_version} · "
        f"config {revision.fingerprint[:12]}…"
    )
    csv_column, json_column = st.columns(2)
    csv_column.download_button(
        "CSV を保存",
        data=csv_bytes(bundle),
        file_name=export_filename(bundle, "csv"),
        mime="text/csv",
        key="download_session_csv",
    )
    json_column.download_button(
        "JSON を保存",
        data=json_bytes(bundle),
        file_name=export_filename(bundle, "json"),
        mime="application/json",
        key="download_session_json",
    )


def _render_advanced_settings(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.2 Advanced 設定. Hidden by default; basic users never see this."""
    with st.expander("Advanced 設定（検出信頼度・角度閾値など）", expanded=False):
        st.caption(
            "既定値からの変更は左右差の計算条件を変えるため、"
            "適用すると履歴・平滑化がリセットされ、新しい比較区間として記録されます。"
        )
        angle_column, gate_column = st.columns(2)
        with angle_column:
            st.markdown("##### 姿勢角の許容範囲")
            yaw_inner = st.slider("yaw inner (deg)", 1.0, 44.0, 15.0, key="adv_yaw_inner")
            yaw_outer = st.slider("yaw outer (deg)", 1.0, 45.0, 17.0, key="adv_yaw_outer")
            pitch_inner = st.slider("pitch inner (deg)", 1.0, 44.0, 15.0, key="adv_pitch_inner")
            pitch_outer = st.slider("pitch outer (deg)", 1.0, 45.0, 17.0, key="adv_pitch_outer")
            roll_inner = st.slider("roll inner (deg)", 1.0, 44.0, 25.0, key="adv_roll_inner")
            roll_outer = st.slider("roll outer (deg)", 1.0, 45.0, 28.0, key="adv_roll_outer")
        with gate_column:
            st.markdown("##### 顔サイズ・安定性")
            face_area_min = st.slider(
                "face_area_min", 0.001, 0.98, 0.02, step=0.001, key="adv_face_area_min"
            )
            face_area_max = st.slider(
                "face_area_max", 0.02, 0.99, 0.55, step=0.01, key="adv_face_area_max"
            )
            valid_frames_to_ok = st.slider(
                "valid_frames_to_ok", 1, 30, 5, key="adv_valid_frames_to_ok"
            )
            ema_tau_seconds = st.slider(
                "ema_tau_seconds", 0.05, 5.0, DEFAULT_TAU_SECONDS, step=0.05, key="adv_ema_tau"
            )

        st.markdown("##### 追跡ズーム")
        zoom_columns = st.columns(5)
        margin_ratio = zoom_columns[0].slider(
            "margin_ratio", 0.0, 3.0, 0.6, step=0.05, key="adv_zoom_margin"
        )
        min_window_size = zoom_columns[1].slider(
            "min_window", 0.01, 1.0, 0.18, step=0.01, key="adv_zoom_min_window"
        )
        recheck_interval_ms = zoom_columns[2].slider(
            "recheck_ms", 0, 5000, 1000, step=100, key="adv_zoom_recheck"
        )
        target_px = zoom_columns[3].slider(
            "target_px", 32, 1920, 480, step=16, key="adv_zoom_target_px"
        )
        max_scale = zoom_columns[4].slider(
            "max_scale", 1.0, 8.0, 4.0, step=0.5, key="adv_zoom_max_scale"
        )

        if st.button("設定を適用", key="apply_advanced_settings"):
            try:
                thresholds = QualityThresholds(
                    yaw_inner_deg=yaw_inner,
                    yaw_outer_deg=yaw_outer,
                    pitch_inner_deg=pitch_inner,
                    pitch_outer_deg=pitch_outer,
                    roll_inner_deg=roll_inner,
                    roll_outer_deg=roll_outer,
                    face_area_min=face_area_min,
                    face_area_max=face_area_max,
                    valid_frames_to_ok=valid_frames_to_ok,
                )
                zoom_policy = TrackingZoomPolicy(
                    margin_ratio=margin_ratio,
                    min_window_size=min_window_size,
                    recheck_interval_ms=recheck_interval_ms,
                    target_px=target_px,
                    max_scale=max_scale,
                )
            except (QualityConfigError, TrackingZoomError) as error:
                st.error(str(error))
            else:
                pipeline.apply_advanced_settings(
                    thresholds=thresholds,
                    tau_seconds=ema_tau_seconds,
                    zoom_policy=zoom_policy,
                )
                st.success("設定を適用し、新しい比較区間を開始しました。")


def render_camera_section() -> None:
    """Render the realtime landmark preview. Call only after consent."""
    st.subheader("リアルタイム ランドマーク表示")
    st.caption(
        "映像は端末内のメモリだけで処理され、保存も外部送信もしません。"
        "プレビューは毎フレーム更新し、ランドマーク推論は最新フレームを最大 8 Hz で抽出します。"
        "状態・左右差・姿勢の数値はカメラ映像の左上にも大きく表示します。"
    )

    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return

    show_midline = st.toggle(
        "顔の中心線を表示",
        value=True,
        key="camera_midline_enabled",
        help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
    )
    pipeline.set_midline_enabled(show_midline)

    brightness_column, background_column = st.columns(2)
    with brightness_column:
        brightness_gain = st.slider(
            "映像の明るさ",
            min_value=MIN_BRIGHTNESS_GAIN,
            max_value=MAX_BRIGHTNESS_GAIN,
            value=1.0,
            step=0.05,
            key="camera_brightness_gain",
            help="暗い環境での検出・見やすさを補助する表示調整です。左右差の計算方法自体は変えません。",
        )
        pipeline.set_brightness_gain(brightness_gain)
    with background_column:
        segmenter_path = configured_segmenter_path()
        if segmenter_status(segmenter_path) is InstallStatus.READY:
            background_mode = st.radio(
                "背景を置き換える（表示専用）",
                options=list(_BACKGROUND_MODE_LABELS),
                format_func=lambda mode: _BACKGROUND_MODE_LABELS[mode],
                key="camera_background_mode",
                horizontal=True,
                help="人物以外を単色にします。左右差・年齢性別・肌表面参考値の判定には使いません。",
            )
            pipeline.set_background_mode(background_mode)
        else:
            st.caption("背景置換モデルが未配置のため、この機能は使用できません。")

    _render_advanced_settings(pipeline)

    preview_column, result_column = st.columns([7, 5], gap="large")
    with preview_column:
        context = webrtc_streamer(
            key=WEBRTC_KEY,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=LOOPBACK_RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_frame_callback=build_video_frame_callback(pipeline),
            async_processing=True,
        )

    with result_column:
        st.markdown("#### ライブ解析")
        if context.state.playing:
            _render_status_panel(pipeline)
        else:
            st.info("`START` を押すと、ブラウザがカメラの許可を求めます。")

    if not context.state.playing:
        return

    _render_export_panel(pipeline)

    st.caption("カメラ映像は `STOP` で停止します。")
    controls = st.columns(2)
    if controls[0].button("スコア・履歴・平滑化をリセット", key="reset_analysis"):
        pipeline.reset_analysis()
        st.rerun()
    if controls[1].button("推論を停止してモデルを解放", key="release_landmarker"):
        release_session_pipeline()
        st.rerun()
