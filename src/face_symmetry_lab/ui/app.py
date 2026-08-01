from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st

from face_symmetry_lab.adapters.inference.onnx_age_estimator import (
    AgeInferenceError,
    OnnxAgeEstimator,
)
from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
)
from face_symmetry_lab.domain.models import AGE_BANDS, AgeEstimate

APP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGE_MODEL = APP_ROOT / "models" / "age_mobilenetv2_utkface_v1.onnx"
AGE_MODEL_VERSION = "age-mobilenetv2-utkface/1.0.0"
ESTIMATOR_STATE_KEY = "age_estimator"
ESTIMATOR_PATH_KEY = "age_estimator_path"
RESULT_STATE_KEY = "age_estimate"


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


def render_app() -> None:
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    st.title("Face Symmetry Lab")
    st.caption("Local processing · 画像非保存 · 研究用")
    st.write(
        "顔の左右差と、モデルが推定した参考年齢帯を端末内で確認するアプリです。"
        "現在の実装マイルストーンでは、年齢帯の静止画推論を先行提供しています。"
    )

    with st.expander("利用前に確認してください", expanded=True):
        consent_privacy = st.checkbox(
            "画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。"
        )
        consent_research = st.checkbox(
            "結果は研究上の参考値であり、実年齢・医学的状態・人物の価値を示しません。"
        )
        consent_rights = st.checkbox(
            "本人の画像、または分析する明示的な権限・同意がある画像だけを使用します。"
        )

    consented = consent_privacy and consent_research and consent_rights
    if not consented:
        st.info("3 項目を確認すると、画像を選択できます。")
        return

    model_path = _configured_model_path()
    if model_path.is_file():
        st.success(f"年齢帯モデル準備済み: {model_path.name}")
    else:
        st.warning(
            "年齢帯モデルが未配置です。"
            "`models/age_mobilenetv2_utkface_v1.onnx` に配置するか、"
            "`FACE_SYMMETRY_LAB_AGE_MODEL_PATH` を設定してください。"
        )

    uploaded = st.file_uploader(
        "単一の顔を中心に切り抜いた JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        help="UTKFace-Cropped と同等の顔 crop を想定しています。上限 10MB / 12MP。",
        on_change=_clear_session_result,
    )
    if uploaded is None:
        return

    try:
        decoded = decode_uploaded_image(uploaded.getvalue())
    except ImageInputError as error:
        st.error(str(error))
        return

    preview_column, result_column = st.columns([7, 5])
    with preview_column:
        st.image(decoded.rgb, caption=f"{decoded.width} × {decoded.height}", width="stretch")
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
            st.session_state[RESULT_STATE_KEY] = estimator.estimate(
                np.asarray(decoded.rgb, dtype=np.uint8)
            )
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
        "左右差の静止画分析は次の M1 実装です。"
        "年齢帯推論とは独立した結果・失敗状態として接続します。"
    )


if __name__ == "__main__":
    render_app()
