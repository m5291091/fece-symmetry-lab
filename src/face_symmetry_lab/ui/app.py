from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeImageLandmarker,
)
from face_symmetry_lab.adapters.inference.onnx_age_estimator import (
    AgeInferenceError,
    OnnxAgeEstimator,
)
from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
)
from face_symmetry_lab.application.still_image import detect_still_image_landmarks
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import AGE_BANDS, AgeEstimate
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.model_setup import landmark_asset, resolve_ready_landmarker

APP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGE_MODEL = APP_ROOT / "models" / "age_mobilenetv2_utkface_v1.onnx"
AGE_MODEL_VERSION = "age-mobilenetv2-utkface/1.0.0"
ESTIMATOR_STATE_KEY = "age_estimator"
ESTIMATOR_PATH_KEY = "age_estimator_path"
RESULT_STATE_KEY = "age_estimate"
INPUT_MODE_STATE_KEY = "input_mode"
INPUT_MODE_CAMERA = "Camera"
INPUT_MODE_IMAGE = "Image"
IMAGE_LANDMARKER_STATE_KEY = "image_landmarker"
IMAGE_LANDMARKER_PATH_KEY = "image_landmarker_path"

_IMAGE_STATUS_MESSAGES = {
    TrackingStatus.NO_FACE: (
        "顔を検出できませんでした。明るさ、顔の大きさ、向きを確認してください。"
    ),
    TrackingStatus.SINGLE_FACE: "1 名の顔を検出しました。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけの写真を使用してください。",
    TrackingStatus.ERROR: "この画像を処理できませんでした。",
}


def _configured_model_path() -> Path:
    configured = os.environ.get("FACE_SYMMETRY_LAB_AGE_MODEL_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_AGE_MODEL


def _session_estimator(model_path: Path) -> OnnxAgeEstimator:
    stat = model_path.stat()
    path_fingerprint = f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    if (
        st.session_state.get(ESTIMATOR_PATH_KEY) != path_fingerprint
        or ESTIMATOR_STATE_KEY not in st.session_state
    ):
        st.session_state[ESTIMATOR_STATE_KEY] = OnnxAgeEstimator(
            model_path=model_path,
            model_version=AGE_MODEL_VERSION,
        )
        st.session_state[ESTIMATOR_PATH_KEY] = path_fingerprint
    estimator = st.session_state[ESTIMATOR_STATE_KEY]
    if not isinstance(estimator, OnnxAgeEstimator):
        raise AgeInferenceError("年齢推定セッションを初期化できません。")
    return estimator


def _clear_session_result() -> None:
    st.session_state.pop(RESULT_STATE_KEY, None)


def _render_age_result(estimate: AgeEstimate) -> None:
    st.subheader("モデル推定の参考年齢帯")
    st.metric("参考年齢帯", estimate.band.value)
    st.caption(
        f"最大クラス確率 {estimate.confidence:.1%}・"
        "実年齢の断定、本人確認、人物評価には使用できません。"
    )
    st.bar_chart(
        {
            "年齢帯": [band.value for band in AGE_BANDS],
            "モデル出力": list(estimate.probabilities),
        },
        x="年齢帯",
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


def _render_image_landmarks(rgb: np.ndarray) -> None:
    """Detect and draw landmarks for one uploaded photo (stateless, §8.7)."""
    st.subheader("静止画のランドマーク")
    model_path = resolve_ready_landmarker()
    if model_path is None:
        return

    try:
        result = detect_still_image_landmarks(_session_image_landmarker(model_path), rgb)
    except LandmarkDetectionError as error:
        st.error(str(error))
        return

    preview_column, status_column = st.columns([7, 5])
    with preview_column:
        st.image(result.preview, caption="検出したランドマーク", width="stretch")
    with status_column:
        st.write(_IMAGE_STATUS_MESSAGES[result.frame.status])
        st.metric("検出した顔の数", result.frame.face_count)
        st.metric("ランドマーク点数", result.landmark_count)
        st.caption("この結果は顔の位置情報であり、左右差スコアや年齢帯とは無関係です。")


def _render_image_section() -> None:
    uploaded = st.file_uploader(
        "JPEG / PNG の顔写真",
        type=["jpg", "jpeg", "png"],
        help="1 名だけが写る写真を想定しています。上限 10MB / 12MP。",
        on_change=_clear_session_result,
    )
    if uploaded is None:
        st.info("写真を選ぶと、ランドマーク検出と参考年齢帯を別々に表示します。")
        return

    try:
        decoded = decode_uploaded_image(uploaded.getvalue())
    except ImageInputError as error:
        st.error(str(error))
        return

    rgb = np.asarray(decoded.rgb, dtype=np.uint8)
    _render_image_landmarks(rgb)

    st.divider()
    st.subheader("モデル推定の参考年齢帯")
    st.caption(f"入力画像 {decoded.width} × {decoded.height}")

    model_path = _configured_model_path()
    if model_path.is_file():
        st.success(f"年齢帯モデル準備済み: {model_path.name}")
    else:
        st.warning(
            "年齢帯モデルが未配置です。"
            "`models/age_mobilenetv2_utkface_v1.onnx` に配置するか、"
            "`FACE_SYMMETRY_LAB_AGE_MODEL_PATH` を設定してください。"
        )

    action_column, result_column = st.columns([7, 5])
    with action_column:
        analyze = st.button(
            "年齢帯を推定",
            type="primary",
            disabled=not model_path.is_file(),
        )
        if st.button("セッション結果をリセット"):
            _clear_session_result()
            st.rerun()

    if analyze:
        try:
            estimator = _session_estimator(model_path)
            st.session_state[RESULT_STATE_KEY] = estimator.estimate(rgb)
        except AgeInferenceError as error:
            _clear_session_result()
            st.error(str(error))

    with result_column:
        estimate = st.session_state.get(RESULT_STATE_KEY)
        if isinstance(estimate, AgeEstimate):
            _render_age_result(estimate)
        else:
            st.info("モデル準備後に「年齢帯を推定」を押すと、8 年代の参考値を表示します。")

    st.divider()
    st.subheader("左右差分析")
    st.info(
        "左右差スコアの算出は次の実装です。"
        "ランドマーク検出・年齢帯推論とは独立した結果・失敗状態として接続します。"
    )


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
    st.title("Face Symmetry Lab")
    st.caption("Local processing · 画像非保存 · 研究用")
    st.write(
        "顔のランドマークと、モデルが推定した参考年齢帯を端末内で確認するアプリです。"
        "現在の実装マイルストーンでは、リアルタイムのランドマーク表示と"
        "静止画の年齢帯推論を提供しています。左右差スコアはまだ算出しません。"
    )

    with st.expander("利用前に確認してください", expanded=True):
        consent_privacy = st.checkbox(
            "映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。"
        )
        consent_research = st.checkbox(
            "結果は研究上の参考値であり、実年齢・医学的状態・人物の価値を示しません。"
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

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        _render_image_section()


if __name__ == "__main__":
    render_app()
