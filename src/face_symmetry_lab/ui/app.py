from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import streamlit as st

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeImageLandmarker,
)
from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import LOCK_PATH, file_sha256
from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
    extract_face_crop,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.application.still_image import (
    STILL_IMAGE_THRESHOLDS,
    StillImageAnalysis,
    analyze_still_image,
)
from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.export import InputMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfig, MetricConfigError
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.symmetrization import SymmetrizationError
from face_symmetry_lab.ui.about_page import render_about_page
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.model_setup import (
    landmark_asset,
    resolve_ready_demographic,
    resolve_ready_landmarker,
)
from face_symmetry_lab.ui.scores import (
    DEVIATION_UNIT_CAPTION,
    format_distance,
    render_demographic_estimate,
    render_headline_metrics,
    render_score_details,
    render_tracking_status,
)

ESTIMATOR_STATE_KEY = "demographic_estimator"
ESTIMATOR_PATH_KEY = "demographic_estimator_paths"
RESULT_STATE_KEY = "demographic_estimate"
INPUT_MODE_STATE_KEY = "input_mode"
#: The last *valid* selection. `st.segmented_control` reports `None` when the
#: active option is clicked again, and §15.1 requires an input to always be
#: active, so the resolved mode is remembered separately from the widget.
ACTIVE_INPUT_MODE_KEY = "active_input_mode"
INPUT_MODE_CAMERA = "Camera"
INPUT_MODE_IMAGE = "Image"

#: FR-004: label, icon, and what the mode actually does, so the choice can be
#: read before it is made. Keyed by the same values the session state holds,
#: so the stored preference survives a wording change.
_INPUT_MODE_CARDS: Final[dict[str, tuple[str, str, str]]] = {
    INPUT_MODE_CAMERA: (
        "Web カメラ",
        ":material/videocam:",
        "映像を見ながら、顔の向きや距離をその場で調整できます。"
        "数値は最大 8 Hz で更新され、映像は端末内のメモリだけで処理されます。",
    ),
    INPUT_MODE_IMAGE: (
        "画像ファイル",
        ":material/image:",
        "JPEG / PNG を 1 枚アップロードして分析します。"
        "10MB・12MP 以下、EXIF の向きを適用し、平滑化のない 1 回きりの結果を表示します。",
    ),
}
IMAGE_LANDMARKER_STATE_KEY = "image_landmarker"
IMAGE_LANDMARKER_PATH_KEY = "image_landmarker_path"
IMAGE_RECORDER_STATE_KEY = "image_session_recorder"
IMAGE_RECORDED_INPUT_KEY = "image_recorded_input"
CONSENT_PRIVACY_KEY = "consent_privacy"
CONSENT_RESEARCH_KEY = "consent_research"
CONSENT_RIGHTS_KEY = "consent_rights"

_IMAGE_STATUS_MESSAGES = {
    TrackingStatus.NO_FACE: (
        "顔を検出できませんでした。明るさ、顔の大きさ、向きを確認してください。"
    ),
    TrackingStatus.SINGLE_FACE: "1 名の顔を検出しました。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけの写真を使用してください。",
    TrackingStatus.ERROR: "この画像を処理できませんでした。",
}


def _session_estimator(
    onnx_path: Path,
    model_version: str,
) -> OpenCvDemographicEstimator:
    onnx_stat = onnx_path.stat()
    path_fingerprint = (
        f"{onnx_path.resolve()}:{onnx_stat.st_size}:{onnx_stat.st_mtime_ns}:{model_version}"
    )
    if (
        st.session_state.get(ESTIMATOR_PATH_KEY) != path_fingerprint
        or ESTIMATOR_STATE_KEY not in st.session_state
    ):
        st.session_state[ESTIMATOR_STATE_KEY] = OpenCvDemographicEstimator(
            onnx_path=onnx_path,
            model_version=model_version,
        )
        st.session_state[ESTIMATOR_PATH_KEY] = path_fingerprint
    estimator = st.session_state[ESTIMATOR_STATE_KEY]
    if not isinstance(estimator, OpenCvDemographicEstimator):
        raise DemographicInferenceError("年齢・性別推定セッションを初期化できません。")
    return estimator


def _clear_session_result() -> None:
    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDER_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDED_INPUT_KEY, None)


def _session_image_landmarker(model_path: Path) -> MediaPipeImageLandmarker:
    """Keep one IMAGE-mode landmarker per session, separate from the video one."""
    fingerprint = str(model_path.resolve())
    if (
        st.session_state.get(IMAGE_LANDMARKER_PATH_KEY) != fingerprint
        or IMAGE_LANDMARKER_STATE_KEY not in st.session_state
    ):
        release_image_landmarker()
        st.session_state[IMAGE_LANDMARKER_STATE_KEY] = MediaPipeImageLandmarker(
            model_path=model_path,
            model_version=landmark_asset().version,
        )
        st.session_state[IMAGE_LANDMARKER_PATH_KEY] = fingerprint
    landmarker = st.session_state[IMAGE_LANDMARKER_STATE_KEY]
    if not isinstance(landmarker, MediaPipeImageLandmarker):
        raise LandmarkDetectionError("ランドマーク推論を初期化できません。")
    return landmarker


def release_image_landmarker() -> None:
    """Release the still-image landmarker (FR-004, §10.4)."""
    landmarker = st.session_state.pop(IMAGE_LANDMARKER_STATE_KEY, None)
    st.session_state.pop(IMAGE_LANDMARKER_PATH_KEY, None)
    if isinstance(landmarker, MediaPipeImageLandmarker):
        landmarker.close()


def _session_image_recorder(
    landmarker: MediaPipeImageLandmarker,
    config: MetricConfig,
) -> SessionRecorder:
    recorder = st.session_state.get(IMAGE_RECORDER_STATE_KEY)
    if not isinstance(recorder, SessionRecorder):
        recorder = SessionRecorder(
            metric_config=config,
            thresholds=STILL_IMAGE_THRESHOLDS,
            tau_seconds=DEFAULT_TAU_SECONDS,
            landmark_model_version=landmarker.model_version,
            landmark_model_sha256=landmarker.model_sha256,
            app_version=__version__,
            lock_digest=file_sha256(LOCK_PATH),
        )
        st.session_state[IMAGE_RECORDER_STATE_KEY] = recorder
    return recorder


def _render_image_export(recorder: SessionRecorder) -> None:
    render_export_panel(
        recorder.export_bundle(),
        heading="この画像の分析データをエクスポート",
        warning=(
            "姿勢・左右差・肌表面参考値・参考年齢・参考性別などの顔由来データを含みます。"
            "共有先と利用目的を確認してから保存してください。"
        ),
        # A single photo is always exactly one record, so a count would add
        # nothing here.
        key_prefix="download_image",
    )


def _analyze_uploaded_image(
    rgb: np.ndarray,
    *,
    input_fingerprint: str,
    model_path: Path,
    show_midline: bool,
) -> tuple[SessionRecorder, StillImageAnalysis]:
    """Detect, score, and record one uploaded photo (stateless, §8.7).

    Draws nothing: the caller owns layout, so a failure here can never take
    an unrelated panel down with it.
    """
    config = load_metric_config()
    landmarker = _session_image_landmarker(model_path)
    started = time.monotonic()
    analysis = analyze_still_image(landmarker, rgb, config=config, show_midline=show_midline)
    processing_ms = (time.monotonic() - started) * 1000.0

    recorder = _session_image_recorder(landmarker, config)
    if st.session_state.get(IMAGE_RECORDED_INPUT_KEY) != input_fingerprint:
        if recorder.records():
            recorder.reset()
        recorder.append_analysis(
            analysis.landmarks.frame,
            analysis.outcome,
            elapsed_ms=0,
            processing_ms=processing_ms,
            input_mode=InputMode.IMAGE,
        )
        st.session_state[IMAGE_RECORDED_INPUT_KEY] = input_fingerprint
    return recorder, analysis


def _render_image_demographic_controls(
    age_input: np.ndarray | None,
    *,
    recorder: SessionRecorder | None,
) -> None:
    """On-demand age/gender trigger for the still-image flow (§10.8).

    Kept beside the photo, next to the headline numbers it fills in
    (§5.1). The runtime caption and full detail panel live in
    `_render_image_demographic_detail`, always full width below.
    """
    demographic_model = resolve_ready_demographic()
    analyze_column, reset_column = st.columns(2)
    analyze = analyze_column.button(
        "参考年齢・参考性別を推定",
        type="primary",
        disabled=demographic_model is None or age_input is None,
        width="stretch",
    )
    if reset_column.button("結果をリセット", width="stretch"):
        _clear_session_result()
        st.rerun()

    if analyze and demographic_model is not None and age_input is not None:
        onnx_path, model_version = demographic_model
        try:
            estimate = _session_estimator(onnx_path, model_version).estimate(age_input)
            st.session_state[RESULT_STATE_KEY] = estimate
            if recorder is not None:
                recorder.attach_demographic_estimate(estimate, observed_elapsed_ms=0)
        except Exception as error:
            # Invariant 12: an age/gender failure must not stop the symmetry
            # side, so nothing beyond this panel may see the exception.
            _clear_session_result()
            message = (
                str(error)
                if isinstance(error, DemographicInferenceError)
                else "年齢・性別を推定できませんでした。"
            )
            st.error(message)


def _render_image_symmetrization(
    rgb: np.ndarray,
    analysis: StillImageAnalysis,
) -> None:
    """FR-034: the same one-shot fold as Camera, on the uploaded photo.

    An upload is already a still frame, so there is nothing to capture — the
    view is computed straight from the photo on screen (§14.3g).
    """
    st.markdown("#### 左右対称にそろえた位置との比較")
    try:
        view = symmetrize_frame(rgb, analysis.landmarks.frame, config=load_metric_config())
    except SymmetrizationError as error:
        st.info(str(error))
        return

    st.caption(
        "写真の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
        "実測点と重ねています。ズレの大きさを形で見るための表示で、"
        "どちらが正しい・望ましいという意味はありません。"
    )
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


def _render_image_demographic_detail(age_input: np.ndarray | None) -> None:
    """Full age/gender reference panel — always full width below (§5.1).

    FR-024 keeps this independent of the symmetry side: it renders even when
    landmark detection failed, explaining what is missing instead of silently
    disappearing.
    """
    st.caption("runtime: OpenCV DNN CPU · Open Model Zoo 2023.0 FP16由来ONNX / 対象18〜75歳")
    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    if isinstance(session_estimate, DemographicEstimate):
        render_demographic_estimate(session_estimate)
    elif age_input is None:
        st.info("単一顔を検出できた写真でのみ、参考年齢・参考性別を推定できます。")
    else:
        st.info("ボタンを押すと、18〜75歳向けの参考値を表示します。")


def _render_image_section() -> None:
    uploaded = st.file_uploader(
        "JPEG / PNG の顔写真",
        type=["jpg", "jpeg", "png"],
        help="1 名だけが写る写真を想定しています。上限 10MB / 12MP。",
        on_change=_clear_session_result,
    )
    if uploaded is None:
        st.info("18〜75歳の写真を選ぶと、左右差と年齢・性別の参考値を写真の横に並べて表示します。")
        return

    uploaded_bytes = uploaded.getvalue()
    try:
        decoded = decode_uploaded_image(uploaded_bytes)
    except ImageInputError as error:
        st.error(str(error))
        return

    rgb = np.asarray(decoded.rgb, dtype=np.uint8)
    st.subheader("静止画のランドマークと左右差")
    model_path = resolve_ready_landmarker()

    recorder: SessionRecorder | None = None
    analysis: StillImageAnalysis | None = None
    if model_path is not None:
        show_midline = st.toggle(
            "顔の中心線を表示",
            value=True,
            key="image_midline_enabled",
            help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
        )
        try:
            recorder, analysis = _analyze_uploaded_image(
                rgb,
                input_fingerprint=sha256(uploaded_bytes).hexdigest(),
                model_path=model_path,
                show_midline=show_midline,
            )
        except (LandmarkDetectionError, MetricConfigError) as error:
            st.error(str(error))

    landmark_frame = analysis.landmarks.frame if analysis is not None else None
    age_input: np.ndarray | None = None
    if landmark_frame is not None:
        try:
            age_input = extract_face_crop(rgb, landmark_frame)
        except ImageInputError:
            age_input = None

    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    estimate = session_estimate if isinstance(session_estimate, DemographicEstimate) else None
    overall = None
    if analysis is not None and analysis.outcome.symmetry is not None:
        overall = analysis.outcome.symmetry.overall

    # §5.1: photo and headline (score/gender/age + status) side by side, so
    # nothing the user needs at a glance lives further down the page. Camera
    # and Image share this layout.
    preview_column, status_column = st.columns([7, 5], gap="large")
    with preview_column:
        if analysis is None:
            st.info("ランドマークを表示できません。")
        else:
            st.image(analysis.landmarks.preview, caption="検出したランドマーク", width="stretch")

    with status_column:
        render_headline_metrics(overall=overall, estimate=estimate)
        st.divider()
        if analysis is not None:
            render_tracking_status(analysis.landmarks.frame.status, _IMAGE_STATUS_MESSAGES)
        _render_image_demographic_controls(age_input, recorder=recorder)

    # §5.1, AC-LAYOUT-01: detail results always render full width below the
    # photo and headline, mirroring the Camera layout.
    st.markdown("#### 詳細結果")
    with st.container(border=True):
        if analysis is not None:
            st.caption(
                f"入力 {rgb.shape[1]} × {rgb.shape[0]} · "
                f"ランドマーク点数 {analysis.landmarks.landmark_count}"
            )
            # Still images are stateless, so the raw value is shown directly (§8.7).
            render_score_details(analysis.outcome, show_smoothed=False)
            st.divider()
            _render_image_symmetrization(rgb, analysis)
            st.divider()
        _render_image_demographic_detail(age_input)

    if recorder is not None:
        _render_image_export(recorder)


def _render_input_mode_selector() -> str:
    """FR-004: pick the input, with each option explained before choosing.

    `st.segmented_control` allows deselecting the active option, which would
    leave the page with no input at all; §15.1 requires one to always be
    active, so a `None` reading falls back to the previous mode rather than
    rendering an empty screen. Selection stays a native widget so keyboard
    operation is unchanged (AC-A11Y-01).
    """
    st.markdown("#### 入力方法")
    # Read from a key of our own, not the widget's: once the widget exists,
    # its key holds `None` for "deselected", which is exactly the value the
    # fallback has to survive.
    previous = st.session_state.get(ACTIVE_INPUT_MODE_KEY, INPUT_MODE_CAMERA)
    selected = st.segmented_control(
        "入力方法",
        options=list(_INPUT_MODE_CARDS),
        format_func=lambda mode: _INPUT_MODE_CARDS[mode][0],
        key=INPUT_MODE_STATE_KEY,
        on_change=_switch_input_mode,
        label_visibility="collapsed",
    )
    mode = selected if selected in _INPUT_MODE_CARDS else previous
    st.session_state[ACTIVE_INPUT_MODE_KEY] = mode

    for column, (name, (label, icon, description)) in zip(
        st.columns(len(_INPUT_MODE_CARDS)),
        _INPUT_MODE_CARDS.items(),
        strict=True,
    ):
        with column, st.container(border=True):
            heading, badge = st.columns([3, 1], vertical_alignment="center")
            heading.markdown(f"{icon} **{label}**")
            if name == mode:
                badge.badge("選択中", color="blue")
            st.caption(description)
    return mode


def _switch_input_mode() -> None:
    """FR-004: drop the previous input's runtime state when the mode changes."""
    release_session_pipeline()
    release_image_landmarker()
    _clear_session_result()


def _render_about_footer() -> None:
    """Summary + link to the About page (§4.1, §5.1).

    The full methodology, error sources, prohibited uses, and model
    versions live on the About page instead of here, so this page stays
    focused on consent → input selection → analysis.
    """
    st.divider()
    st.subheader("Face Symmetry Lab について")
    st.write(
        "顔のランドマーク、撮影補正後の肌表面参考値、モデル推定の参考年齢・参考性別を端末内で確認します。"
        "リアルタイム映像にはランドマークと中心線を重ね、スコアは映像の隣に表示します。"
        "数値履歴は再現性メタデータ付きの CSV / JSON として保存できます。"
    )
    boundary, metric, cadence = st.columns(3)
    boundary.metric("処理境界", "端末内のみ")
    metric.metric("左右差指標", "相対比較専用")
    cadence.metric("カメラ解析", "最大 8 Hz")
    st.info(
        "左右差は両眼の間隔に対する対応点のずれとして表示します（0〜100 のスコアではありません）。"
        "数値は、同一人物・同一環境での相対変化だけを参照してください。"
    )
    st.page_link(ABOUT_PAGE, label="対称度の算出方法・制約について詳しく見る", icon="📘")


def render_app() -> None:
    """The Analyze page: consent → input selection → analysis (§5.1)."""
    st.title("Face Symmetry Lab")
    st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")

    # Collapsed once already agreed to (from the prior run's widget state), so
    # returning users are not forced to scroll past it on every rerun.
    already_consented = all(
        st.session_state.get(key, False)
        for key in (CONSENT_PRIVACY_KEY, CONSENT_RESEARCH_KEY, CONSENT_RIGHTS_KEY)
    )
    with st.expander("利用前に確認してください", expanded=not already_consented):
        consent_privacy = st.checkbox(
            "映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。",
            key=CONSENT_PRIVACY_KEY,
        )
        consent_research = st.checkbox(
            "私は18〜75歳です。結果は参考値であり、実年齢、自己認識・法的性別、"
            "肌質・医学的状態、人物の価値を示さないことに同意します。",
            key=CONSENT_RESEARCH_KEY,
        )
        consent_rights = st.checkbox(
            "本人の映像・画像、または分析する明示的な権限・同意があるものだけを使用します。",
            key=CONSENT_RIGHTS_KEY,
        )

    consented = consent_privacy and consent_research and consent_rights
    if not consented:
        st.info("3 項目を確認すると、カメラまたは画像を選択できます。")
        return

    input_mode = _render_input_mode_selector()
    st.divider()

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        _render_image_section()

    _render_about_footer()


#: Streamlit `st.navigation` multipage entry points (§4.1). Constructed at
#: import time — `st.Page` does not require a running script context — so
#: `render_app` can reference `ABOUT_PAGE` for its `st.page_link` regardless
#: of how the module is entered.
ANALYZE_PAGE = st.Page(render_app, title="Analyze", icon="🪞", default=True)
ABOUT_PAGE = st.Page(render_about_page, title="使い方・算出方法", icon="📘")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    st.navigation([ANALYZE_PAGE, ABOUT_PAGE]).run()
