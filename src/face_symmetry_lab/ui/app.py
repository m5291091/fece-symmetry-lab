"""Analyze ページの構成と、Streamlit の multipage 入口（§4.1, §5.1）。

**ここが持つのはページの骨格だけである**——同意ゲート、入力方法の選択、選ばれた
モードへの委譲、末尾の About 導線。入力モードそれぞれの中身は `ui.camera`
（Camera）と `ui.image_section`（Image）が持つ。以前ここに Image モード一式が
同居していたため、ページ構成と片方のモードが同じ file にあり、2 つのモードが
非対称に見えていた。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.ui.about_page import render_about_page
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.image_section import (
    clear_session_result,
    release_image_landmarker,
    render_image_section,
)
from face_symmetry_lab.ui.reference_page import render_reference_page

INPUT_MODE_STATE_KEY = "input_mode"
#: 最後に *有効だった* 選択。`st.segmented_control` は選択中の項目をもう一度
#: 押すと `None` を報告するが、§15.1 は常にどれかの入力が有効であることを求める。
#: そのため解決済みのモードを widget とは別に覚えておく。
ACTIVE_INPUT_MODE_KEY = "active_input_mode"
INPUT_MODE_CAMERA = "Camera"
INPUT_MODE_IMAGE = "Image"

CONSENT_PRIVACY_KEY = "consent_privacy"
CONSENT_RESEARCH_KEY = "consent_research"
CONSENT_RIGHTS_KEY = "consent_rights"

#: FR-004: ラベル・アイコン・そのモードが実際に何をするか。選ぶ前に読めるように
#: するためである。セッション状態が持つ値と同じキーを使うので、文言を変えても
#: 保存済みの設定は生き残る。
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


def _switch_input_mode() -> None:
    """FR-004: モードが変わったら、直前の入力の runtime 状態を破棄する。"""
    release_session_pipeline()
    release_image_landmarker()
    clear_session_result()


def _render_input_mode_selector() -> str:
    """FR-004: 入力方法を選ぶ。各選択肢は選ぶ前に説明を読めるようにする。

    `st.segmented_control` は選択中の項目を解除でき、そのままではページに入力
    が 1 つも無い状態になる。§15.1 は常にどれかが有効であることを求めるため、
    `None` を読んだときは空の画面を描かずに直前のモードへ戻す。選択自体は
    ネイティブ widget のままなので、キーボード操作は変わらない（AC-A11Y-01）。
    """
    st.markdown("#### 入力方法")
    # widget のキーではなく自前のキーから読む。widget が生成された後、その
    # キーは「解除」を意味する `None` を保持することがあり、それこそが
    # fallback が乗り越えなければならない値だからである。
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


def _render_consent_gate() -> bool:
    """不変条件 3: 画像 upload と camera permission の前に明示同意を取る。

    すでに同意済みなら（直前の実行の widget 状態から判断して）畳んでおく。
    再訪した利用者が毎回この節をスクロールで通り過ぎずに済むようにする。
    """
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
    return consent_privacy and consent_research and consent_rights


def _render_about_footer() -> None:
    """概要と About ページへのリンク（§4.1, §5.1）。

    算出方法の全体、誤差要因、禁止用途、モデルの版は、ここではなく About
    ページに置く。このページを「同意 → 入力選択 → 分析」に集中させるためである。
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
    """Analyze ページ: 同意 → 入力選択 → 分析（§5.1）。"""
    st.title("Face Symmetry Lab")
    st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")

    if not _render_consent_gate():
        st.info("3 項目を確認すると、カメラまたは画像を選択できます。")
        return

    input_mode = _render_input_mode_selector()
    st.divider()

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        render_image_section()

    _render_about_footer()


#: Streamlit `st.navigation` の multipage 入口（§4.1）。import 時に構築する。
#: `st.Page` は実行中のスクリプトコンテキストを必要としないため、モジュールの
#: 入り方に関わらず `render_app` が `st.page_link` から `ABOUT_PAGE` を参照
#: できる。
ANALYZE_PAGE = st.Page(render_app, title="Analyze", icon="🪞", default=True)
REFERENCE_PAGE = st.Page(render_reference_page, title="参照統計", icon="📊")
ABOUT_PAGE = st.Page(render_about_page, title="使い方・算出方法", icon="📘")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    st.navigation([ANALYZE_PAGE, REFERENCE_PAGE, ABOUT_PAGE]).run()
