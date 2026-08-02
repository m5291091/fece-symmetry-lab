"""Streamlit + streamlit-webrtc composition for realtime landmark preview.

Composition only: this module wires the MediaPipe adapter to the realtime use
case and renders the result. No geometry, no inference, and no model download
happens here. The WebRTC callback runs on a worker thread and therefore never
touches Streamlit state (§14.3).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import av
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, webrtc_streamer

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeFaceLandmarker,
)
from face_symmetry_lab.adapters.inference.mediapipe_segmenter import (
    MediaPipeSegmenter,
    SegmentationError,
)
from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.adapters.inference.tracking_zoom_landmarker import (
    TrackingZoomLandmarker,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import InstallStatus, file_sha256
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.demographic_worker import DemographicWorker
from face_symmetry_lab.application.realtime import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.guidance import guidance_message
from face_symmetry_lab.ui.model_setup import (
    configured_demographic_path,
    configured_segmenter_path,
    demographic_asset,
    demographic_model_status,
    landmark_asset,
    resolve_ready_landmarker,
    segmenter_asset,
    segmenter_status,
)
from face_symmetry_lab.ui.scores import (
    render_demographic_estimate,
    render_headline_metrics,
    render_score_details,
)

_BACKGROUND_MODE_LABELS: Final[dict[BackgroundMode, str]] = {
    BackgroundMode.OFF: "しない",
    BackgroundMode.BLACK: "黒",
    BackgroundMode.WHITE: "白",
}

#: FR-032: display-only placement preset. Does not affect scoring.
_PANEL_LAYOUT_RIGHT: Final[str] = "動画の右"
_PANEL_LAYOUT_LEFT: Final[str] = "動画の左"
_PANEL_LAYOUT_BELOW: Final[str] = "動画の下"
_PANEL_LAYOUT_OPTIONS: Final[tuple[str, ...]] = (
    _PANEL_LAYOUT_RIGHT,
    _PANEL_LAYOUT_LEFT,
    _PANEL_LAYOUT_BELOW,
)

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
    """Adapt the realtime pipeline to the streamlit-webrtc frame callback.

    FR-013: the pipeline draws only landmarks and the midline (~1ms per
    frame). The numeric panel is deliberately not drawn here — rendering it
    cost 18.6ms of the 33ms available per frame at 720p/30fps and was what
    surfaced as lag — so the numbers live in the panel beside the video
    instead. When nothing changed the pixels, the frame that arrived is
    handed straight back with no re-encode.
    """

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        try:
            rgb = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
            painted, _ = pipeline.process_with_analysis(rgb)
            if painted is rgb:
                # Untouched: skip re-encoding it into a new VideoFrame.
                return frame
            replaced = av.VideoFrame.from_ndarray(painted, format="rgb24")
        except Exception:
            # Returning the source frame keeps the preview alive (NFR-017).
            return frame
        if frame.pts is not None:
            replaced.pts = frame.pts
        if frame.time_base is not None:
            replaced.time_base = frame.time_base
        return replaced

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


def _build_demographic_worker() -> DemographicWorker | None:
    """Age/gender inference is optional (§14.3, FR-024): never blocks the camera.

    Uses the silent status query rather than `resolve_ready_demographic`,
    which renders its own status block — a builder must not paint UI.
    """
    path = configured_demographic_path()
    if demographic_model_status(path) is not InstallStatus.READY:
        return None
    try:
        estimator = OpenCvDemographicEstimator(
            onnx_path=path,
            model_version=demographic_asset().version,
        )
    except DemographicInferenceError:
        return None
    return DemographicWorker(estimator=estimator)


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
        demographic_worker=_build_demographic_worker(),
        analysis_interval_ms=DEFAULT_ANALYSIS_INTERVAL_MS,
        midline_indices=config.midline_indices,
        # §15.2: Advanced settings have no manual UI. brightness/zoom/cadence
        # are AI auto-tuned unconditionally (FR-029, FR-030); quality-gate
        # thresholds and part_weights stay fixed at their constructor
        # defaults above and are never exposed for editing.
        auto_tune_enabled=True,
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
def _render_headline_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """Poll the immutable snapshot at 5 Hz for the headline block only.

    FR-013/FR-032: total score, gender, age, tracking status and position
    guidance stay beside the video (placement follows the panel_layout
    preset) so the user can watch them without looking away from the
    camera. Everything else is detail and lives in `_render_detail_fragment`
    instead, always full width below (§5.1, AC-LAYOUT-01).
    """
    snapshot = pipeline.snapshot()
    with st.container(border=True):
        symmetry = snapshot.analysis.symmetry if snapshot.analysis is not None else None
        overall = snapshot.analysis.overall_smooth if snapshot.analysis is not None else None
        if overall is None and symmetry is not None:
            overall = symmetry.overall
        estimate = snapshot.demographic_estimate
        render_headline_metrics(
            overall=overall,
            gender=None if estimate is None else estimate.gender,
            age_years=None if estimate is None else estimate.estimated_age_years,
        )
        st.divider()
        message = _STATUS_MESSAGES[snapshot.status]
        if snapshot.status is TrackingStatus.ERROR:
            st.error(message)
        elif snapshot.status is TrackingStatus.MULTI_FACE:
            st.warning(message)
        elif snapshot.status is TrackingStatus.SINGLE_FACE:
            st.success(message)
        else:
            st.info(message)

        if snapshot.analysis is not None:
            guidance = guidance_message(snapshot.analysis.state)
            if guidance is not None:
                st.warning(guidance)
        if snapshot.needs_recovery:
            st.warning("処理が続けて失敗しています。カメラを停止して再開してください。")


def _render_headline_panel(
    pipeline: RealtimeLandmarkPipeline,
    context: WebRtcStreamerContext[Any, Any],
) -> None:
    """The "ライブ解析" heading + headline block, placed per panel_layout (FR-032)."""
    st.markdown("#### ライブ解析")
    if context.state.playing:
        _render_headline_fragment(pipeline)
    else:
        st.info("`START` を押すと、ブラウザがカメラの許可を求めます。")


@st.fragment(run_every=STATUS_REFRESH_INTERVAL)
def _render_detail_fragment(pipeline: RealtimeLandmarkPipeline) -> None:
    """Poll the immutable snapshot at 5 Hz for the full-width detail block."""
    snapshot = pipeline.snapshot()
    with st.container(border=True):
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
        # §14.3f: a NFR-002 diagnostic breakdown of the total above, so a slow
        # session can be traced to a stage before considering GPU/deployment
        # changes. segmentation/scoring only show once that stage has run.
        breakdown = [f"landmark {snapshot.landmark_ms:.1f} ms"]
        if snapshot.segmentation_ms > 0.0:
            breakdown.append(f"背景置換 {snapshot.segmentation_ms:.1f} ms")
        if snapshot.scoring_ms > 0.0:
            breakdown.append(f"scoring {snapshot.scoring_ms:.1f} ms")
        st.caption("内訳: " + " · ".join(breakdown))

        if snapshot.analysis is not None:
            st.divider()
            st.caption("カメラ映像の LAST SAMPLE と同じ直近の解析値です。")
            render_score_details(snapshot.analysis)

        if snapshot.demographic_estimate is not None:
            st.divider()
            # No chart: this fragment repaints at 5 Hz (§11 NFR-003).
            render_demographic_estimate(
                snapshot.demographic_estimate,
                show_distribution=False,
            )


def _render_detail_panel(
    pipeline: RealtimeLandmarkPipeline,
    context: WebRtcStreamerContext[Any, Any],
) -> None:
    """部位別・姿勢・肌表面参考値などの詳細結果。配置プリセットに関わらず常に

    動画とヘッダー結果の下へ画面幅いっぱいで表示する（FR-032, AC-LAYOUT-01）。
    """
    if not context.state.playing:
        return
    st.markdown("#### 詳細結果")
    _render_detail_fragment(pipeline)


def _render_export_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    render_export_panel(
        pipeline.export_bundle(),
        heading="セッションデータのエクスポート",
        warning=(
            "姿勢・左右差などの顔由来の派生データを含みます。"
            "共有先と利用目的を確認してから保存してください。"
        ),
        key_prefix="download_session",
        # A live session accumulates records, so the count tells the user
        # how much this download actually covers.
        show_record_count=True,
    )


def _render_auto_tune_status(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.2: Advanced settings have no manual UI; show what the AI picked.

    Quality-gate thresholds and `part_weights` are fixed constants and are
    intentionally not shown here as if they were tunable (§14.3d).
    """
    zoom_policy = pipeline.zoom_policy
    zoom_caption = f" · ズーム目標 {zoom_policy.target_px}px" if zoom_policy is not None else ""
    st.caption(
        "Advanced設定はAIが自動調整します（明るさ・追跡ズーム・"
        "history_capacity・age_interval_ms・analysis_interval_ms・sensitivity）。"
        f"現在値: 検出用の明るさ補正 ×{pipeline.brightness_gain:.2f}{zoom_caption} · "
        f"analysis_interval_ms={pipeline.analysis_interval_ms}。"
        "明るさ補正はlandmark検出にだけ使い、表示される映像の明るさは変えません。"
        "品質ゲート閾値と部位の重み(part_weights)は固定既定値です。"
    )


def render_camera_section() -> None:
    """Render the realtime landmark preview. Call only after consent."""
    st.subheader("リアルタイム ランドマーク表示")
    st.caption(
        "映像は端末内のメモリだけで処理され、保存も外部送信もしません。"
        "プレビューは毎フレーム更新し、ランドマーク推論は最新フレームを最大 8 Hz で抽出します。"
        "映像にはランドマークと中心線だけを重ね、全体スコア・参考性別・参考年齢は映像のすぐ隣に大きく表示します。"
        "部位別・姿勢・肌表面参考値などの詳しい数値は、その下に画面幅いっぱいで表示します。"
    )

    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return

    midline_column, demographic_column = st.columns(2)
    with midline_column:
        show_midline = st.toggle(
            "顔の中心線を表示",
            value=True,
            key="camera_midline_enabled",
            help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
        )
    with demographic_column:
        # Gated on the worker this pipeline actually holds, so the toggle can
        # never claim the feature is live while no worker exists.
        show_demographics = st.toggle(
            "年齢・性別の参考値を表示",
            value=True,
            key="camera_demographic_enabled",
            disabled=not pipeline.has_demographic_worker,
            help=(
                "18〜75歳向けモデル推定の参考年齢・参考性別を最大1Hzで独立推論します。"
                "左右差の計算とは分離しており、失敗してもカメラ映像は止まりません。"
            ),
        )
        if not pipeline.has_demographic_worker:
            st.caption("年齢・性別モデルが未配置のため使用できません。")
    pipeline.set_midline_enabled(show_midline)
    pipeline.set_demographic_enabled(show_demographics)

    background_column, status_column = st.columns(2)
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
    with status_column:
        _render_auto_tune_status(pipeline)

    # FR-032: display-only placement preset ("movable" via preset choice, not
    # mouse drag — see §14.3e). Does not change scoring.
    panel_layout = st.radio(
        "「ライブ解析」パネルの配置",
        options=_PANEL_LAYOUT_OPTIONS,
        key="camera_panel_layout",
        horizontal=True,
    )

    result_column: DeltaGenerator | None
    if panel_layout == _PANEL_LAYOUT_BELOW:
        preview_column = st.container()
        result_column = None
    elif panel_layout == _PANEL_LAYOUT_LEFT:
        result_column, preview_column = st.columns([5, 7], gap="large")
    else:
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

    if result_column is None:
        _render_headline_panel(pipeline, context)
    else:
        with result_column:
            _render_headline_panel(pipeline, context)

    # §5.1, AC-LAYOUT-01: detail results always render full width below the
    # video and headline, regardless of the panel_layout preset above.
    _render_detail_panel(pipeline, context)

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
