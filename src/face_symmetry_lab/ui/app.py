from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import streamlit as st

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.export.session_exporter import (
    csv_bytes,
    export_filename,
    json_bytes,
)
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeImageLandmarker,
)
from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.adapters.io.metric_config import load_metric_config
from face_symmetry_lab.adapters.io.model_store import file_sha256
from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
    extract_face_crop,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.application.still_image import (
    STILL_IMAGE_THRESHOLDS,
    analyze_still_image,
)
from face_symmetry_lab.domain.export import InputMode
from face_symmetry_lab.domain.landmarks import LandmarkFrame, TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfig, MetricConfigError
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.model_setup import (
    landmark_asset,
    resolve_ready_demographic,
    resolve_ready_landmarker,
)
from face_symmetry_lab.ui.scores import render_scores

APP_ROOT = Path(__file__).resolve().parents[3]
ESTIMATOR_STATE_KEY = "demographic_estimator"
ESTIMATOR_PATH_KEY = "demographic_estimator_paths"
RESULT_STATE_KEY = "demographic_estimate"
INPUT_MODE_STATE_KEY = "input_mode"
INPUT_MODE_CAMERA = "Camera"
INPUT_MODE_IMAGE = "Image"
IMAGE_LANDMARKER_STATE_KEY = "image_landmarker"
IMAGE_LANDMARKER_PATH_KEY = "image_landmarker_path"
IMAGE_RECORDER_STATE_KEY = "image_session_recorder"
IMAGE_RECORDED_INPUT_KEY = "image_recorded_input"

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


def _render_demographic_result(estimate: DemographicEstimate) -> None:
    st.subheader("モデル推定の参考年齢・参考性別")
    age_column, gender_column = st.columns(2)
    age_column.metric("参考年齢", f"{estimate.estimated_age_years:.1f}歳")
    gender_label = {GenderLabel.FEMALE: "女性", GenderLabel.MALE: "男性"}[estimate.gender]
    gender_column.metric("外見由来の参考性別", gender_label)
    st.caption(
        f"性別モデル確率 {estimate.gender_confidence:.1%}・対象18〜75歳。"
        "年齢や本人の自己認識・法的性別を断定せず、本人確認や人物評価には使用できません。"
    )
    st.bar_chart(
        {
            "モデルラベル": ["female", "male"],
            "モデル出力": list(estimate.gender_probabilities),
        },
        x="モデルラベル",
        y="モデル出力",
    )
    with st.expander("モデル情報"):
        st.code(
            f"version: {estimate.model_version}\n"
            f"sha256: {estimate.model_sha256}\n"
            f"observed_at: {estimate.observed_at.isoformat()}",
            language="text",
        )


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
            lock_digest=file_sha256(APP_ROOT / "uv.lock"),
        )
        st.session_state[IMAGE_RECORDER_STATE_KEY] = recorder
    return recorder


def _render_image_export(recorder: SessionRecorder) -> None:
    bundle = recorder.export_bundle()
    if not bundle.records:
        return
    st.divider()
    st.subheader("この画像の分析データをエクスポート")
    st.warning(
        "姿勢・左右差・肌表面参考値・参考年齢・参考性別などの顔由来データを含みます。"
        "共有先と利用目的を確認してから保存してください。"
    )
    revision = bundle.manifest.config_revisions[-1]
    st.caption(f"metric {bundle.manifest.metric_version} · config {revision.fingerprint[:12]}…")
    csv_column, json_column = st.columns(2)
    csv_column.download_button(
        "CSV を保存",
        data=csv_bytes(bundle),
        file_name=export_filename(bundle, "csv"),
        mime="text/csv",
        key="download_image_csv",
    )
    json_column.download_button(
        "JSON を保存",
        data=json_bytes(bundle),
        file_name=export_filename(bundle, "json"),
        mime="application/json",
        key="download_image_json",
    )


def _render_image_landmarks(
    rgb: np.ndarray,
    *,
    input_fingerprint: str,
) -> tuple[SessionRecorder | None, LandmarkFrame | None]:
    """Detect, draw, and score one uploaded photo (stateless, §8.7)."""
    st.subheader("静止画のランドマークと左右差")
    model_path = resolve_ready_landmarker()
    if model_path is None:
        return None, None

    show_midline = st.toggle(
        "顔の中心線を表示",
        value=True,
        key="image_midline_enabled",
        help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
    )

    try:
        config = load_metric_config()
        landmarker = _session_image_landmarker(model_path)
        started = time.monotonic()
        analysis = analyze_still_image(
            landmarker,
            rgb,
            config=config,
            show_midline=show_midline,
        )
        processing_ms = (time.monotonic() - started) * 1000.0
    except (LandmarkDetectionError, MetricConfigError) as error:
        st.error(str(error))
        return None, None

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

    landmarks = analysis.landmarks
    preview_column, status_column = st.columns([7, 5])
    with preview_column:
        st.image(landmarks.preview, caption="検出したランドマーク", width="stretch")
        st.caption(_IMAGE_STATUS_MESSAGES[landmarks.frame.status])
        st.caption(f"ランドマーク点数 {landmarks.landmark_count}")
    with status_column:
        # Still images are stateless, so the raw value is shown directly (§8.7).
        render_scores(analysis.outcome, show_smoothed=False)
    return recorder, landmarks.frame


def _render_image_section() -> None:
    uploaded = st.file_uploader(
        "JPEG / PNG の顔写真",
        type=["jpg", "jpeg", "png"],
        help="1 名だけが写る写真を想定しています。上限 10MB / 12MP。",
        on_change=_clear_session_result,
    )
    if uploaded is None:
        st.info("18〜75歳の写真を選ぶと、左右差と年齢・性別の参考値を別々に表示します。")
        return

    uploaded_bytes = uploaded.getvalue()
    try:
        decoded = decode_uploaded_image(uploaded_bytes)
    except ImageInputError as error:
        st.error(str(error))
        return

    rgb = np.asarray(decoded.rgb, dtype=np.uint8)
    recorder, landmark_frame = _render_image_landmarks(
        rgb,
        input_fingerprint=sha256(uploaded_bytes).hexdigest(),
    )

    st.divider()
    st.subheader("モデル推定の参考年齢・参考性別")
    st.caption(f"入力画像 {decoded.width} × {decoded.height}")
    st.caption("runtime: OpenCV DNN CPU · Open Model Zoo 2023.0 FP16由来ONNX / 対象18〜75歳")

    age_input: np.ndarray | None = None
    if landmark_frame is not None:
        try:
            age_input = extract_face_crop(rgb, landmark_frame)
        except ImageInputError:
            age_input = None

    demographic_model = resolve_ready_demographic()
    model_ready = demographic_model is not None

    action_column, result_column = st.columns([7, 5])
    with action_column:
        analyze = st.button(
            "参考年齢・参考性別を推定",
            type="primary",
            disabled=not model_ready or age_input is None,
        )
        if st.button("セッション結果をリセット"):
            _clear_session_result()
            st.rerun()

    if analyze:
        try:
            if demographic_model is None:
                raise DemographicInferenceError("検証済みの年齢・性別モデルがありません。")
            onnx_path, model_version = demographic_model
            estimator = _session_estimator(onnx_path, model_version)
            if age_input is None:
                raise DemographicInferenceError("単一顔のcropを作成できません。")
            estimate = estimator.estimate(age_input)
            st.session_state[RESULT_STATE_KEY] = estimate
            if recorder is not None:
                recorder.attach_demographic_estimate(estimate, observed_elapsed_ms=0)
        except DemographicInferenceError as error:
            _clear_session_result()
            st.error(str(error))

    with result_column:
        session_estimate = st.session_state.get(RESULT_STATE_KEY)
        if isinstance(session_estimate, DemographicEstimate):
            _render_demographic_result(session_estimate)
        else:
            st.info("モデル準備後にボタンを押すと、18〜75歳向けの参考値を表示します。")

    if recorder is not None:
        _render_image_export(recorder)


def _switch_input_mode() -> None:
    """FR-004: drop the previous input's runtime state when the mode changes."""
    release_session_pipeline()
    release_image_landmarker()
    _clear_session_result()


def render_app() -> None:
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    with st.container(border=True):
        st.title("Face Symmetry Lab")
        st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")
        st.write(
            "顔のランドマーク、撮影補正後の肌表面参考値、モデル推定の参考年齢・参考性別を端末内で確認します。"
            "リアルタイム映像には左右差スコアと姿勢角を重ね、"
            "数値履歴は再現性メタデータ付きの CSV / JSON として保存できます。"
        )
        boundary, metric, cadence = st.columns(3)
        boundary.metric("処理境界", "端末内のみ")
        metric.metric("左右差指標", "相対比較専用")
        cadence.metric("カメラ解析", "最大 8 Hz")

    st.info(
        "感度係数の固定評価セット校正は未完了です。"
        "左右差の数値は、同一人物・同一環境での相対変化だけを参照してください。"
    )

    with st.expander("スコアの算出技術と精度について"):
        st.markdown(
            "- **幾何左右差**: MediaPipe Face Landmarkerの478点を、両眼で回転・大きさ・"
            "位置を正規化します。正中点へTLS直線を当て、右側を鏡映した後、輪郭・目・"
            "鼻・口の対応点RMS距離を0〜100へ変換します。\n"
            "- **肌表面参考値（実験値）**: 顔cropのwhite balanceと明度を補正し、頬patchの"
            "色調・局所texture均一性を比較します。幾何の全体点には混ぜません。\n"
            "- **主な誤差要因**: 顔の向き、照明、影、レンズ歪み、ランドマーク誤差、"
            "化粧、髭、眼鏡、カメラ処理です。\n"
            "- **精度向上**: 固定評価セットでの係数校正、実写pose符号確認、3D正規化比較、"
            "照明・ぼけgate、同一端末での再撮影評価が必要です。"
        )

    with st.expander("利用前に確認してください", expanded=True):
        consent_privacy = st.checkbox(
            "映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。"
        )
        consent_research = st.checkbox(
            "私は18〜75歳です。結果は参考値であり、実年齢、自己認識・法的性別、"
            "肌質・医学的状態、人物の価値を示さないことに同意します。"
        )
        consent_rights = st.checkbox(
            "本人の映像・画像、または分析する明示的な権限・同意があるものだけを使用します。"
        )

    consented = consent_privacy and consent_research and consent_rights
    if not consented:
        st.info("3 項目を確認すると、カメラまたは画像を選択できます。")
        return

    input_mode = st.radio(
        "入力方法",
        options=(INPUT_MODE_CAMERA, INPUT_MODE_IMAGE),
        horizontal=True,
        key=INPUT_MODE_STATE_KEY,
        on_change=_switch_input_mode,
    )
    st.divider()

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        _render_image_section()


if __name__ == "__main__":
    render_app()
