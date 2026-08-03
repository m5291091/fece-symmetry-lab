"""Streamlit + streamlit-webrtc composition for realtime landmark preview.

Composition only: this module wires the MediaPipe adapter to the realtime use
case and renders the result. No geometry, no inference, and no model download
happens here. The WebRTC callback runs on a worker thread and therefore never
touches Streamlit state (§14.3).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import av
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit.delta_generator import DeltaGenerator
from streamlit_webrtc import WebRtcMode, WebRtcStreamerContext, webrtc_streamer

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.imaging.piecewise_warp import OpenCvPiecewiseWarper
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
from face_symmetry_lab.adapters.io.model_store import LOCK_PATH, InstallStatus, file_sha256
from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.demographic_worker import DemographicWorker
from face_symmetry_lab.application.realtime import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    PipelineSnapshot,
    RealtimeLandmarkPipeline,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.frame_adjustments import BackgroundMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfigError
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.symmetrization import SymmetrizationError
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
    DEVIATION_UNIT_CAPTION,
    GENDER_LABELS,
    format_distance,
    format_symmetry,
    gate_color,
    render_demographic_estimate,
    render_headline_metrics,
    render_score_details,
    render_symmetrized_image,
    render_tracking_status,
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
PANEL_LAYOUT_STATE_KEY = "camera_panel_layout"

#: §14.3e/FR-032: the HUD moves by preset, like the panel layout, rather than
#: by dragging. "非表示" draws nothing at all.
_HUD_HIDDEN: Final[str] = "非表示"
_HUD_POSITION_OPTIONS: Final[tuple[str, ...]] = ("左上", "右上", "左下", "右下", _HUD_HIDDEN)
HUD_POSITION_STATE_KEY = "camera_hud_position"
FULLSCREEN_STATE_KEY = "camera_fullscreen"
DEMOGRAPHIC_ENABLED_STATE_KEY = "camera_demographic_enabled"

#: Where each preset pins the HUD inside its positioning context.
_HUD_PLACEMENTS: Final[dict[str, str]] = {
    "左上": "top:24px; left:24px;",
    "右上": "top:24px; right:24px;",
    "左下": "bottom:24px; left:24px;",
    "右下": "bottom:24px; right:24px;",
}

#: §14.3e: the fullscreen band's wording. Kept as constants so the copy can be
#: asserted on — this is the only text shown while the page chrome is folded
#: away, so it has to carry both the state and the way out.
_FULLSCREEN_BADGE: Final[str] = "全画面表示中 — 右上の「終了」で戻ります"
_FULLSCREEN_HINT: Final[str] = (
    "画面下の「START」を押すとカメラが始まります。"
    "ブラウザの F11 と併用すると、タブやアドレスバーも隠れた完全な全画面になります。"
)

#: The hidden element the 5 Hz fragment writes and the bridge script reads.
#: Values travel as JSON text: Streamlit sanitizes markdown HTML, and text
#: content is the one thing guaranteed to survive it intact.
_HUD_DATA_ELEMENT_ID: Final[str] = "fsl-hud-data"

#: §5.2 exception (§14.3e): the stylesheet is installed **once** by the bridge
#: and thereafter only toggled by a class, never re-injected. Re-sending a
#: `<style>` at 5 Hz forces a whole-document style recalculation five times a
#: second, which is what made the video stutter; the same goes for
#: `backdrop-filter`, which re-blurs the region behind the HUD on *every*
#: video frame rather than on every update. Neither is used any more.
#:
#: The video is found through `.st-key-camera_preview_frame iframe` — the hook
#: `st.container(key=...)` emits — so nothing here depends on a Streamlit
#: internal `data-testid`, which is what silently broke fullscreen before.
_HUD_STYLESHEET: Final[str] = """
#fsl-hud {
    position: fixed; z-index: 1000050;
    background: rgba(0,0,0,.72); color: #fff;
    padding: 16px 22px; border-radius: 16px;
    border: 2px solid #6B6B6B;
    pointer-events: none;
    font-variant-numeric: tabular-nums;
    font-family: inherit;
}
#fsl-hud[hidden] { display: none; }
/* Fullscreen state band: says where you are and, before START, what to do.
   Centred so it cannot collide with the HUD (top-left) or the exit control
   (top-right). */
#fsl-chrome {
    position: fixed; top: 24px; left: 50%; transform: translateX(-50%);
    z-index: 1000055; pointer-events: none;
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    font-family: inherit; text-align: center;
}
#fsl-chrome[hidden] { display: none; }
#fsl-chrome-badge {
    background: rgba(0,0,0,.72); color: #fff;
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: 6px 16px; font-size: .8rem; letter-spacing: .04em;
}
#fsl-chrome-hint {
    background: rgba(0,0,0,.72); color: #fff;
    border-radius: 12px; padding: 10px 18px; font-size: 1rem; max-width: 70vw;
}
#fsl-chrome-hint[hidden] { display: none; }
#fsl-hud-value { font-size: 3.2rem; font-weight: 700; line-height: 1; }
#fsl-hud-label { font-size: .85rem; opacity: .75; margin-bottom: 10px; }
#fsl-hud-demo { font-size: 1.1rem; }
#fsl-hud-demo:empty { display: none; }
/* Only the grip takes pointer events, so the video's own STOP button stays
   clickable everywhere else under the HUD. */
#fsl-hud-grip {
    pointer-events: auto; cursor: move; touch-action: none;
    margin-top: 10px; font-size: .75rem; opacity: .55;
    border-top: 1px solid rgba(255,255,255,.25); padding-top: 6px;
}
.st-key-camera_preview_frame { position: relative; border-width: 3px !important; }
/* `width: auto` is load-bearing: Streamlit's vertical block is width:100%,
   so a fixed container anchored by `right` would stretch across the viewport
   and drop its button off the left edge. */
.st-key-fsl_exit_fullscreen {
    position: fixed; top: 24px; right: 24px; left: auto; z-index: 1000060;
    width: auto !important; min-width: 0 !important;
}
.st-key-fsl_exit_fullscreen button {
    background: rgba(0,0,0,.72) !important;
    color: #fff !important;
    border: 2px solid rgba(255,255,255,.65) !important;
    padding: 10px 22px !important;
    font-size: 1rem !important;
}
body.fsl-fullscreen .st-key-camera_preview_frame iframe {
    position: fixed; inset: 0;
    width: 100vw !important; height: 100vh !important;
    z-index: 1000000; background: #000;
}
/* Streamlit's own chrome sits at z-index 999990/999991 — far above any value
   a page-level overlay would naturally pick — so fullscreen has to stack
   above it and fold the chrome away, or the sidebar covers a third of the
   video and the header hides the HUD. `.stSidebar`/`.stAppHeader` are
   Streamlit's semantic class names, not emotion hashes, so they are stable
   hooks (§5.2). Verified against streamlit 1.59.1. */
body.fsl-fullscreen .stSidebar,
body.fsl-fullscreen .stAppHeader { display: none !important; }
"""

#: §10.2 forbids STUN and TURN. An empty ICE list keeps the session on loopback.
LOOPBACK_RTC_CONFIGURATION: Final[dict[str, list[object]]] = {"iceServers": []}

#: streamlit-webrtc's own `video_html_attrs`, which is the supported way to
#: configure the `<video>` — far steadier than reaching into the component's
#: iframe. Two things matter here:
#:
#: * `controls: False` removes the native control bar entirely. Its fullscreen
#:   button promoted the `<video>` into the browser's top layer, above every
#:   page element including the HUD; and play/pause/seek mean nothing for a
#:   live preview. START/STOP are the component's own buttons, not these.
#: * The default is `style={"width": "100%"}` with *auto* height, which is why
#:   the picture stayed letterboxed at the top of a full-height frame instead
#:   of filling the screen. Fullscreen sizing is finished in `_VIDEO_CHROME_CSS`.
VIDEO_HTML_ATTRS: Final[dict[str, object]] = {
    "autoPlay": True,
    "controls": False,
    "muted": True,
    "playsInline": True,
    "controlsList": "nofullscreen noremoteplayback nodownload",
    "disablePictureInPicture": True,
    "style": {"width": "100%", "background": "#000"},
}

#: FR-005: video only. No audio permission is requested.
MEDIA_STREAM_CONSTRAINTS: Final[dict[str, object]] = {
    "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
    "audio": False,
}

WEBRTC_KEY = "face-landmark-preview"
PIPELINE_STATE_KEY = "landmark_pipeline"

#: FR-034: the symmetrization view for the frame the user captured. Session
#: state only — never written to disk, and cleared with the analysis (§10.4).
SYMMETRIZATION_STATE_KEY = "symmetrization_view"

#: §11 NFR-003: the result panel refreshes at 4-5 Hz, independent of video fps.
STATUS_REFRESH_INTERVAL = "200ms"

#: The bridge's own DOM-only refresh. Same cadence as the panels, but it
#: costs no server round trip — it just re-reads the hidden payload.
_HUD_REFRESH_MS: Final[int] = 200

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
        # thresholds stay fixed at their constructor defaults above and are
        # never exposed for editing. Nothing auto-tuned touches scoring.
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
    st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)
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
        overall = (
            snapshot.analysis.overall_distance_smooth if snapshot.analysis is not None else None
        )
        if overall is None and symmetry is not None:
            overall = symmetry.overall
        render_headline_metrics(overall=overall, estimate=snapshot.demographic_estimate)
        st.divider()
        render_tracking_status(snapshot.status, _STATUS_MESSAGES)

        if snapshot.analysis is not None:
            guidance = guidance_message(snapshot.analysis.state)
            if guidance is not None:
                st.warning(guidance)
        if snapshot.needs_recovery:
            st.warning("処理が続けて失敗しています。カメラを停止して再開してください。")

    # §14.3e: rides along in this existing 5 Hz fragment rather than adding a
    # third one. The element is hidden and found by id, so where it lands in
    # the DOM does not matter.
    _render_hud_payload(snapshot)


def _hud_payload(
    *,
    overall: float | None,
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
) -> dict[str, object]:
    """The §14.3e HUD's values. Pure: no Streamlit, no markup, no side effects.

    Every number goes through the same formatter the panels use
    (`format_symmetry`, `GENDER_LABELS`), so the HUD can never drift into a
    second way of rendering the same measurement. Missing values read `—`
    rather than a stale or zeroed number (NFR-013).

    ``demo`` is empty — not `—` — when there is no estimate or the user turned
    the demographic panel off, since an empty row would imply the feature is
    running and failing (FR-024).
    """
    visible = position != _HUD_HIDDEN and position in _HUD_PLACEMENTS
    demo = ""
    if show_demographics and estimate is not None:
        age = f"{estimate.estimated_age_years:.0f} 歳"
        # `DemographicEstimate` validates that `gender` is the argmax of
        # `gender_probabilities`, so this is that label without re-deriving it.
        gender = GENDER_LABELS.get(estimate.gender, "—")
        demo = f"参考年齢 {age} · 参考性別 {gender}"
    return {
        "visible": visible,
        "symmetry": format_symmetry(overall),
        "label": "対称度（実験値）",
        "demo": demo,
        "gate": gate,
    }


def _hud_html(
    *,
    overall: float | None,
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
) -> str:
    """Serialize the payload into the hidden element the bridge reads.

    Deliberately tiny and unstyled. This is the only thing that crosses the
    wire at 5 Hz, and the previous version — a fully styled block plus a
    `<style>` tag — is what made the video stutter: every repaint forced the
    browser into a whole-document style recalculation.
    """
    payload = _hud_payload(
        overall=overall,
        estimate=estimate,
        show_demographics=show_demographics,
        gate=gate,
        position=position,
    )
    encoded = json.dumps(payload, ensure_ascii=False)
    return f'<div id="{_HUD_DATA_ELEMENT_ID}" style="display:none">{encoded}</div>'


def _render_hud_payload(snapshot: PipelineSnapshot) -> None:
    """Publish one snapshot's HUD values (called from the 5 Hz headline fragment).

    Hidden and located by id, so its position in the DOM does not matter —
    which is what lets it ride along in an existing fragment instead of
    needing a third one of its own.
    """
    analysis = snapshot.analysis
    overall = analysis.overall_distance_smooth if analysis is not None else None
    if overall is None and analysis is not None and analysis.symmetry is not None:
        overall = analysis.symmetry.overall

    st.markdown(
        _hud_html(
            overall=overall,
            estimate=snapshot.demographic_estimate,
            show_demographics=bool(st.session_state.get(DEMOGRAPHIC_ENABLED_STATE_KEY, True)),
            gate=gate_color(analysis.state if analysis is not None else None),
            position=st.session_state.get(HUD_POSITION_STATE_KEY, _HUD_POSITION_OPTIONS[0]),
        ),
        unsafe_allow_html=True,
    )


#: Injected into the WebRTC component's own iframe (same-origin), because a
#: stylesheet in the page does not reach inside it. Only two jobs remain now
#: that `VIDEO_HTML_ATTRS` handles the native controls:
#:
#: * Make the picture actually fill the screen. The component sizes the video
#:   `width: 100%` with auto height, so in a full-height frame it stays a
#:   letterbox strip at the top — pinning it to the frame's viewport is what
#:   fills the screen.
#: * Move the START/STOP row off the far-left edge, where it is genuinely hard
#:   to hit once the frame spans the display. `.fsl-controls` is a class this
#:   app puts on the row from JS, since the component only ships emotion/MUI
#:   hashes and offers no stable hook of its own (§5.2).
_VIDEO_CHROME_CSS: Final[str] = """
html.fsl-fs, html.fsl-fs body { margin: 0; background: #000; }
html.fsl-fs video {
    position: fixed; inset: 0;
    width: 100vw !important; height: 100vh !important;
    object-fit: contain; background: #000;
}
/* Centred with auto margins rather than `translateX(-50%)`: a transform makes
   the element a containing block for *fixed* descendants, and when the row
   turned out to wrap the <video>, the picture was positioned against this pill
   instead of the viewport — which is how it ended up tucked into the corner. */
html.fsl-fs .fsl-controls {
    position: fixed; left: 0; right: 0; bottom: 44px;
    width: max-content; margin-left: auto; margin-right: auto;
    transform: none;
    z-index: 20; gap: 16px;
    background: rgba(0,0,0,.66); padding: 12px 20px; border-radius: 999px;
    box-shadow: 0 6px 24px rgba(0,0,0,.45);
}
html.fsl-fs .fsl-controls button {
    font-size: 1.05rem !important;
    padding: 10px 26px !important;
    min-height: 48px;
}
"""


#: The HUD runtime, injected into the *page's* JS realm (not the component
#: iframe's). Streamlit tears the component iframe down on every rerun, and a
#: timer callback defined inside it dies with its realm while the timer id
#: lives on — which is exactly how the HUD ended up frozen. Running here, the
#: element, its drag handlers and its timer all outlive any rerun.
_HUD_RUNTIME_JS: Final[str] = """
(function () {
  var CFG = window.__fslHudConfig;
  if (!CFG) return;

  var sheet = document.getElementById('fsl-hud-style');
  if (!sheet) {
    sheet = document.createElement('style');
    sheet.id = 'fsl-hud-style';
    document.head.appendChild(sheet);
  }
  if (sheet.textContent !== CFG.css) sheet.textContent = CFG.css;

  var chrome = document.getElementById('fsl-chrome');
  if (!chrome) {
    chrome = document.createElement('div');
    chrome.id = 'fsl-chrome';
    chrome.innerHTML =
      '<span id="fsl-chrome-badge"></span>' +
      '<span id="fsl-chrome-hint"></span>';
    document.body.appendChild(chrome);
  }
  document.getElementById('fsl-chrome-badge').textContent = CFG.badgeText;
  document.getElementById('fsl-chrome-hint').textContent = CFG.hintText;
  chrome.hidden = !CFG.fullscreen;

  var hud = document.getElementById('fsl-hud');
  if (!hud) {
    hud = document.createElement('div');
    hud.id = 'fsl-hud';
    hud.hidden = true;
    hud.innerHTML =
      '<div id="fsl-hud-value"></div>' +
      '<div id="fsl-hud-label"></div>' +
      '<div id="fsl-hud-demo"></div>' +
      '<div id="fsl-hud-grip">\u283f ドラッグで移動</div>';
    document.body.appendChild(hud);

    // Drag by the grip only, so the rest of the HUD stays click-through and
    // the video's own STOP button underneath keeps working.
    var grip = hud.querySelector('#fsl-hud-grip');
    var dragging = false, offsetX = 0, offsetY = 0;
    grip.addEventListener('pointerdown', function (event) {
      var box = hud.getBoundingClientRect();
      dragging = true;
      offsetX = event.clientX - box.left;
      offsetY = event.clientY - box.top;
      try { grip.setPointerCapture(event.pointerId); } catch (err) {}
      event.preventDefault();
    });
    grip.addEventListener('pointermove', function (event) {
      if (!dragging) return;
      var x = Math.max(0, event.clientX - offsetX);
      var y = Math.max(0, event.clientY - offsetY);
      hud.style.left = x + 'px';
      hud.style.top = y + 'px';
      hud.style.right = 'auto';
      hud.style.bottom = 'auto';
      try {
        localStorage.setItem('fslHudPos', JSON.stringify({x: x, y: y}));
      } catch (err) { /* private mode: the position simply is not remembered */ }
    });
    var stop = function (event) {
      if (!dragging) return;
      dragging = false;
      try { grip.releasePointerCapture(event.pointerId); } catch (err) {}
    };
    grip.addEventListener('pointerup', stop);
    grip.addEventListener('pointercancel', stop);
  }

  // Choosing a preset clears any dragged position, so the radio stays a way
  // back to a known corner.
  var applyAnchor = function () { hud.setAttribute('style', CFG.anchor); };
  if (window.__fslHudPreset !== CFG.position) {
    window.__fslHudPreset = CFG.position;
    applyAnchor();
    try { localStorage.removeItem('fslHudPos'); } catch (err) {}
  } else {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem('fslHudPos')); } catch (err) {}
    if (saved && typeof saved.x === 'number') {
      hud.setAttribute('style', 'left:' + saved.x + 'px; top:' + saved.y + 'px;');
    } else {
      applyAnchor();
    }
  }

  document.body.classList.toggle('fsl-fullscreen', !!CFG.fullscreen);

  // One timer, replaced rather than stacked when the bridge re-runs.
  if (window.__fslHudTimer) clearInterval(window.__fslHudTimer);
  // The component iframe is same-origin, so its <video> and its START/STOP
  // row can be reached. Idempotent: safe to re-run on every tick.
  var tameVideoFrame = function () {
    var host = document.querySelector('.st-key-camera_preview_frame iframe');
    var vdoc = null;
    try { vdoc = host && host.contentDocument; } catch (err) { return; }
    if (!vdoc || !vdoc.head) return;
    var vstyle = vdoc.getElementById('fsl-video-style');
    if (!vstyle) {
      vstyle = vdoc.createElement('style');
      vstyle.id = 'fsl-video-style';
      vdoc.head.appendChild(vstyle);
    }
    if (vstyle.textContent !== CFG.videoCss) vstyle.textContent = CFG.videoCss;
    vdoc.documentElement.classList.toggle('fsl-fs', !!CFG.fullscreen);
    // The control row must never be an ancestor of the <video>: styling it
    // would then reposition the picture too. Walk up only as far as the last
    // wrapper that still excludes the video, and drop the class from anything
    // that has since grown to contain it (the tree changes once playing).
    var stale = vdoc.querySelectorAll('.fsl-controls');
    for (var s = 0; s < stale.length; s++) {
      if (stale[s].querySelector('video')) stale[s].classList.remove('fsl-controls');
    }
    var anyButton = vdoc.querySelector('button');
    if (anyButton) {
      var row = null;
      for (var n = anyButton.parentElement; n && n !== vdoc.body; n = n.parentElement) {
        if (n.querySelector('video')) break;
        row = n;
      }
      if (row && !row.classList.contains('fsl-controls')) row.classList.add('fsl-controls');
    }
  };

  var lastGate = '';
  var sync = function () {
    tameVideoFrame();
    var holder = document.getElementById(CFG.dataId);
    var hudEl = document.getElementById('fsl-hud');
    if (!hudEl) return;
    // No payload means the camera is not running: the panel that publishes it
    // only renders while playing, so the HUD must go with it rather than
    // linger showing the last frame's numbers.
    var hint = document.getElementById('fsl-chrome-hint');
    if (!holder) {
      hudEl.hidden = true;
      if (hint) hint.hidden = false;
      return;
    }
    if (hint) hint.hidden = true;
    var data;
    try { data = JSON.parse(holder.textContent); } catch (err) { return; }
    hudEl.hidden = !data.visible;
    if (!data.visible) return;
    var valueEl = document.getElementById('fsl-hud-value');
    var labelEl = document.getElementById('fsl-hud-label');
    var demoEl = document.getElementById('fsl-hud-demo');
    if (valueEl && valueEl.textContent !== data.symmetry) valueEl.textContent = data.symmetry;
    if (labelEl && labelEl.textContent !== data.label) labelEl.textContent = data.label;
    if (demoEl && demoEl.textContent !== data.demo) demoEl.textContent = data.demo;
    if (data.gate !== lastGate) {
      lastGate = data.gate;
      hudEl.style.borderColor = data.gate;
      var frame = document.querySelector('.st-key-camera_preview_frame');
      if (frame) frame.style.borderColor = data.gate;
    }
  };
  sync();
  window.__fslHudTimer = setInterval(sync, CFG.intervalMs);
})();
"""


def _hud_bridge_script(*, fullscreen: bool, position: str) -> str:
    """The loader that installs `_HUD_RUNTIME_JS` into the page.

    Runs inside a `components.v1.html` iframe, which is same-origin with the
    page, so it can reach `window.parent`. It does as little as possible
    there: publish the config, then hand the actual runtime to the parent
    realm as a `<script>`, because anything defined in the iframe stops
    working the moment Streamlit replaces it on the next rerun.

    Attaching the HUD to the page `<body>` is also what makes it survive
    fullscreen: on `<body>` no Streamlit container can clip it or become its
    containing block, so `position: fixed` behaves.
    """
    config = json.dumps(
        {
            "fullscreen": fullscreen,
            "position": position,
            "anchor": _HUD_PLACEMENTS.get(position, ""),
            "dataId": _HUD_DATA_ELEMENT_ID,
            "css": _HUD_STYLESHEET,
            "videoCss": _VIDEO_CHROME_CSS,
            "intervalMs": _HUD_REFRESH_MS,
            "badgeText": _FULLSCREEN_BADGE,
            "hintText": _FULLSCREEN_HINT,
        },
        ensure_ascii=False,
    )
    runtime = json.dumps(_HUD_RUNTIME_JS)
    return f"""
<script>
(function () {{
  var win = window.parent;
  var doc = win && win.document;
  if (!doc || !doc.body) return;
  win.__fslHudConfig = {config};
  var previous = doc.getElementById('fsl-hud-runtime');
  if (previous) previous.remove();
  var runtime = doc.createElement('script');
  runtime.id = 'fsl-hud-runtime';
  runtime.textContent = {runtime};
  doc.head.appendChild(runtime);
}})();
</script>
"""


def _render_hud_bridge(*, fullscreen: bool, position: str) -> None:
    """Install the HUD bridge once per run (never inside a 5 Hz fragment)."""
    components.html(_hud_bridge_script(fullscreen=fullscreen, position=position), height=0)


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
        # The measured result leads; the runtime counters below are diagnostics
        # and were burying it under frame bookkeeping the user cannot act on.
        if snapshot.analysis is not None:
            render_score_details(snapshot.analysis)
        else:
            st.info("品質ゲートを満たすと、ここに部位別の数値が表示されます。")

        if snapshot.demographic_estimate is not None:
            st.divider()
            # No chart: this fragment repaints at 5 Hz (§11 NFR-003).
            render_demographic_estimate(
                snapshot.demographic_estimate,
                show_distribution=False,
            )

        st.divider()
        with st.expander("処理の状態（診断用）"):
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
            # §14.3f: a NFR-002 diagnostic breakdown of the total above, so a
            # slow session can be traced to a stage before considering
            # GPU/deployment changes. segmentation/scoring only show once that
            # stage has run.
            breakdown = [f"landmark {snapshot.landmark_ms:.1f} ms"]
            if snapshot.segmentation_ms > 0.0:
                breakdown.append(f"背景置換 {snapshot.segmentation_ms:.1f} ms")
            if snapshot.scoring_ms > 0.0:
                breakdown.append(f"scoring {snapshot.scoring_ms:.1f} ms")
            st.caption("内訳: " + " · ".join(breakdown))
            _render_auto_tune_status(pipeline)


def _render_detail_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """部位別・姿勢・肌表面参考値などの詳細結果。配置プリセットに関わらず常に

    動画とヘッダー結果の下へ画面幅いっぱいで表示する（FR-032, AC-LAYOUT-01）。
    """
    _render_detail_fragment(pipeline)


def _clear_symmetrization(pipeline: RealtimeLandmarkPipeline) -> None:
    """Drop the captured frame and its view together (§10.4, FR-034)."""
    pipeline.clear_capture()
    st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)


def _consume_pending_capture(pipeline: RealtimeLandmarkPipeline, *, warp: bool) -> None:
    """Turn a fulfilled capture into a view, once, off the realtime path.

    The pipeline hands the frame over exactly once, so this runs the one-shot
    symmetrization and keeps only the result. A capture that cannot be
    symmetrized is reported rather than retried silently.

    ``warp`` requests §8.6.4's image, which rebuilds every pixel and is
    therefore opt-in; it is read at capture time, since the raw frame is
    consumed here and not retained afterwards (§10.4).
    """
    captured = pipeline.take_capture()
    if captured is None:
        return
    config = pipeline.metric_config
    if config is None:
        st.session_state[SYMMETRIZATION_STATE_KEY] = "指標設定が読み込まれていません。"
        return
    try:
        st.session_state[SYMMETRIZATION_STATE_KEY] = symmetrize_frame(
            captured.rgb_image,
            captured.frame,
            config=config,
            warper=OpenCvPiecewiseWarper() if warp else None,
        )
    except SymmetrizationError as error:
        st.session_state[SYMMETRIZATION_STATE_KEY] = str(error)


def _render_symmetrization_panel(pipeline: RealtimeLandmarkPipeline) -> None:
    """FR-034: capture one moment and show it folded about its own midline.

    Deliberately outside the 5 Hz fragments: this is a still result computed
    once from one frame, and the live preview keeps running untouched while
    it is displayed (§14.3g).
    """
    st.markdown("#### 左右対称にそろえた位置との比較")
    st.caption(
        "この瞬間の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
        "実測点と重ねて表示します。ズレの大きさを形で見るための表示で、"
        "どちらが正しい・望ましいという意味はありません。"
    )

    # Read before the button so the toggle's current state is the one that
    # applies to the capture it triggers.
    warp = st.checkbox(
        "左右対称にそろえた画像も生成する（幾何変形・処理が重くなります）",
        key="camera_symmetrization_warp",
        help=(
            "ランドマークを結んだ三角形ごとに画素を変形し、左右対称にそろえた顔画像を作ります。"
            "加工画像であり、無加工の写真ではありません。"
        ),
    )

    _consume_pending_capture(pipeline, warp=warp)
    controls = st.columns(2)
    if controls[0].button("この瞬間を対称化して見る", key="capture_symmetrization"):
        pipeline.request_capture()
        st.rerun()
    if controls[1].button("表示を消す", key="clear_symmetrization"):
        _clear_symmetrization(pipeline)
        st.rerun()

    view = st.session_state.get(SYMMETRIZATION_STATE_KEY)
    if view is None:
        if pipeline.capture_pending:
            st.info("次に顔を検出したフレームを取り込みます。そのままお待ちください。")
        return
    if isinstance(view, str):
        st.warning(view)
        return

    st.image(
        view.overlay,
        caption="実測点と、左右対称にそろえた位置（色違い）を線で結んだ表示",
        width="stretch",
    )
    mean_column, max_column = st.columns(2)
    mean_column.metric("平均のズレ", format_distance(view.mean_deviation))
    max_column.metric("最大のズレ", format_distance(view.max_deviation))
    st.caption(
        f"対称化した点 {view.symmetrized_point_count} 点。"
        "対応する左右のペアが定義されていない点は、実測のまま動かしていません。"
    )
    st.caption(DEVIATION_UNIT_CAPTION)
    render_symmetrized_image(view, requested=warp)
    st.caption("取り込んだ映像はこのブラウザセッションのメモリだけに残り、保存も送信もしません。")


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

    Quality-gate thresholds are fixed constants and are intentionally not
    shown here as if they were tunable (§14.3d). Nothing listed affects a
    measured value: §8.6 left the metric with no coefficient to tune.
    """
    zoom_policy = pipeline.zoom_policy
    zoom_caption = f" · ズーム目標 {zoom_policy.target_px}px" if zoom_policy is not None else ""
    st.caption(
        "Advanced設定はAIが自動調整します（明るさ・追跡ズーム・"
        "history_capacity・age_interval_ms・analysis_interval_ms）。"
        f"現在値: 検出用の明るさ補正 ×{pipeline.brightness_gain:.2f}{zoom_caption} · "
        f"analysis_interval_ms={pipeline.analysis_interval_ms}。"
        "明るさ補正はlandmark検出にだけ使い、表示される映像の明るさは変えません。"
        "品質ゲート閾値は固定既定値です。左右差の計算には調整可能な係数が無いため、"
        "自動調整が数値を動かすことはありません。"
    )


def _render_display_controls(pipeline: RealtimeLandmarkPipeline) -> None:
    """§15.1's display-only preferences, pushed straight onto the pipeline.

    None of these change a score (FR-027, FR-028): the midline and background
    are drawn after analysis, and the age/gender toggle only gates a panel
    that is already independent of the symmetry side (FR-024). Reading them
    here and writing them through in one place keeps that claim checkable.
    """
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

    # §5.4/§14.3e: the border colour is an identifier, not a verdict, and the
    # same information stays available as text in the panel beside the video.
    st.caption(
        "映像の枠色と HUD の枠線は品質ゲートの状態を示します。"
        "緑は「計測条件を満たしている」という意味で、顔の良し悪しを表すものではありません。"
        "同じ状態は映像の隣のテキストでも確認できます。"
    )

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


def _split_for_panel_layout() -> tuple[DeltaGenerator, DeltaGenerator | None]:
    """Return (preview, headline) columns for the stored FR-032 preset.

    Reads the preference instead of rendering its widget, because §5.1 puts
    the video at the very top while the control that changes it lives in the
    "表示設定" tab below. The widget writes the same session-state key, so a
    change is picked up on the rerun it triggers.

    Display-only, and "movable" by preset choice rather than mouse drag
    (§14.3e). A ``None`` headline column means "stack it below the video",
    which is the one preset that needs no column of its own. §5.1's 7:5
    ratio holds whichever side the headline lands on.
    """
    panel_layout = st.session_state.get(PANEL_LAYOUT_STATE_KEY, _PANEL_LAYOUT_RIGHT)
    if panel_layout == _PANEL_LAYOUT_BELOW:
        return st.container(), None
    if panel_layout == _PANEL_LAYOUT_LEFT:
        result_column, preview_column = st.columns([5, 7], gap="large")
        return preview_column, result_column
    preview_column, result_column = st.columns([7, 5], gap="large")
    return preview_column, result_column


def _render_panel_layout_control() -> None:
    """The FR-032 preset widget, rendered inside the "表示設定" tab.

    `st.tabs` builds every tab's contents on each run, so this widget exists
    (and keeps its value) whichever tab is on screen.
    """
    st.radio(
        "「ライブ解析」パネルの配置",
        options=_PANEL_LAYOUT_OPTIONS,
        key=PANEL_LAYOUT_STATE_KEY,
        horizontal=True,
        help="見出し（対称度・参考性別・参考年齢・状態）の位置だけを変えます。数値は変わりません。",
    )
    st.radio(
        "映像上の HUD の位置",
        options=_HUD_POSITION_OPTIONS,
        key=HUD_POSITION_STATE_KEY,
        horizontal=True,
        help="映像へ重ねる対称度・参考年齢・参考性別の位置です。映像自体は加工しません。",
    )


def _exit_fullscreen() -> None:
    """Leave fullscreen from the pinned button.

    A callback rather than an inline assignment: Streamlit refuses to modify
    a widget's session-state key after that widget has been instantiated, and
    the toggle is created earlier in the same run. `on_click` runs before the
    next run builds it, so the write lands legally.
    """
    st.session_state[FULLSCREEN_STATE_KEY] = False


def _render_session_controls(pipeline: RealtimeLandmarkPipeline) -> None:
    """Stop/reset controls, kept reachable from the main screen (§5.1)."""
    st.caption("カメラ映像は映像上の `STOP` で停止します。")
    controls = st.columns(2)
    if controls[0].button("結果・履歴・平滑化をリセット", key="reset_analysis"):
        pipeline.reset_analysis()
        # `reset_analysis` drops the captured frame; the view derived from it
        # has to go with it, or the panel would outlive its own segment.
        st.session_state.pop(SYMMETRIZATION_STATE_KEY, None)
        st.rerun()
    if controls[1].button("推論を停止してモデルを解放", key="release_landmarker"):
        release_session_pipeline()
        st.rerun()


def render_camera_section() -> None:
    """Render the realtime landmark preview. Call only after consent.

    §5.1: the video and its headline come first and stay first. Everything
    else — settings, detail numbers, symmetrization, export — sits below in
    tabs, so only one of them occupies height at a time and the preview
    remains on screen instead of being pushed out of view. Tab switching is
    display-only: it changes no measured value and does not interrupt the
    WebRTC stream (AC-LAYOUT-01).
    """
    st.subheader("リアルタイム ランドマーク表示")
    st.caption(
        "映像は端末内のメモリだけで処理され、保存も外部送信もしません。"
        "映像にはランドマークと中心線だけを重ね、全体の対称度・参考性別・参考年齢は"
        "映像のすぐ隣に大きく表示します。設定と詳しい数値は映像の下のタブにまとめています。"
    )

    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        pipeline = _session_pipeline(model_path)
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return

    fullscreen = st.toggle(
        "映像を全画面表示",
        key=FULLSCREEN_STATE_KEY,
        help=(
            "映像をブラウザ画面いっぱいに広げます。"
            "ブラウザの F11 と併用すると完全な全画面になります。"
        ),
    )
    # §5.2's overlay exception: installed once per run, never inside a
    # fragment, so the 5 Hz update never re-sends CSS.
    _render_hud_bridge(
        fullscreen=fullscreen,
        position=st.session_state.get(HUD_POSITION_STATE_KEY, _HUD_POSITION_OPTIONS[0]),
    )
    if fullscreen:
        # The tabs below are not rendered while fullscreen, so this is the
        # only way back — it has to exist before anything can cover it.
        with st.container(key="fsl_exit_fullscreen"):
            st.button("✕ 全画面を終了", key="exit_fullscreen", on_click=_exit_fullscreen)

    preview_column, result_column = _split_for_panel_layout()

    with preview_column, st.container(border=True, key="camera_preview_frame"):
        context = webrtc_streamer(
            key=WEBRTC_KEY,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=LOOPBACK_RTC_CONFIGURATION,
            media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
            video_frame_callback=build_video_frame_callback(pipeline),
            video_html_attrs=VIDEO_HTML_ATTRS,
            async_processing=True,
        )

    if result_column is None:
        _render_headline_panel(pipeline, context)
    else:
        with result_column:
            _render_headline_panel(pipeline, context)

    if not context.state.playing:
        # Settings still belong on screen before starting, but there is no
        # result, capture, or export to offer yet.
        with st.expander("表示設定"):
            _render_display_controls(pipeline)
            _render_panel_layout_control()
        return

    if fullscreen:
        # Nothing below the video is reachable while it covers the viewport,
        # so skip it entirely rather than leaving 5 Hz fragments repainting
        # panels no one can see.
        st.caption("全画面表示中です。設定・詳細結果は「全画面を終了」で戻ると表示されます。")
        return

    # §5.1, AC-LAYOUT-01: everything below the video is full width and one at
    # a time, so the preview above never scrolls out of reach.
    detail_tab, symmetry_tab, settings_tab, export_tab = st.tabs(
        ["詳細結果", "左右対称との比較", "表示設定", "エクスポート"]
    )
    with detail_tab:
        _render_detail_panel(pipeline)
    with symmetry_tab:
        # FR-034/§14.3g: a still, one-shot view. The live preview above keeps
        # running at its own cadence while this is on screen.
        _render_symmetrization_panel(pipeline)
    with settings_tab:
        _render_display_controls(pipeline)
        _render_panel_layout_control()
        st.divider()
        _render_session_controls(pipeline)
    with export_tab:
        _render_export_panel(pipeline)
