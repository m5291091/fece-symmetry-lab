"""Analyze ページの構成と、Streamlit の multipage 入口（§4.1, §5.1）。

**ここが持つのはページの骨格だけである**——同意ゲート、入力方法の選択、選ばれた
モードへの委譲。入力モードそれぞれの中身は `ui.camera`（Camera）と
`ui.image_section`（Image）が持つ。About・参照統計への導線はページ最下部の
要約ごと削除し（項目 31、H8、2026-08-29）、サイドバーの multipage nav だけが
入口になっている。
"""

from __future__ import annotations

from typing import Final

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from face_symmetry_lab.ui.about_page import render_about_page
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.i18n import render_language_selector, t
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

#: FR-004（2026-08-29 の改訂）: ラベル・アイコン・そのモードが何をするか。
#: **説明を常時表示しない。** 「Webカメラ」「画像アップロード」というラベル
#: だけで何が起きるかは読み取れており、以前ここにあった枠付きの説明カードは
#: Analyze ページで最も縦を占める文の壁になっていた（§5.3.1、項目 30）。説明は
#: `help` のツールチップへ移す——選ぶ前に読めるという FR-004 の要求は、その場で
#: 開ける `help` でも満たせる。セッション状態が持つ値と同じキーを使うので、
#: 文言を変えても保存済みの設定は生き残る。
#:
#: **ここは日本語の原文のまま置く。`t()` を通さない。** この dict は module-level
#: で一度だけ評価されるので、ここで訳すと import した時点の言語で固定され、
#: あとから言語を切り替えてもこの 2 語だけ古い言語のまま残る（FR-043）。訳は
#: 描画のたびに `_input_mode_label` / `_input_mode_help` が通す。
_INPUT_MODES: Final[dict[str, tuple[str, str, str]]] = {
    INPUT_MODE_CAMERA: (
        "Webカメラ",
        ":material/videocam:",
        "映像を見ながら、顔の向きや距離をその場で調整できます。"
        "数値は最大 8 Hz で更新され、映像は端末内のメモリだけで処理されます。",
    ),
    INPUT_MODE_IMAGE: (
        "画像アップロード",
        ":material/image:",
        "JPEG / PNG を 1 枚アップロードして分析します。"
        "10MB・12MP 以下、EXIF の向きを適用し、平滑化のない 1 回きりの結果を表示します。",
    ),
}


def _input_mode_label(mode: str) -> str:
    """選択肢のラベル。アイコンは token なので訳さない（FR-043）。

    分岐を明示するのは `t()` の引数をリテラルに保つためである
    （`t(_INPUT_MODES[mode][0])` は入れ子の添字になり、`tests/ui/test_i18n.py`
    の走査が原文を解決できない）。`ui.camera` の `_background_mode_label` と
    同じ形にそろえてある。
    """
    icon = _INPUT_MODES[mode][1]
    if mode == INPUT_MODE_CAMERA:
        return f"{icon} {t('Webカメラ')}"
    return f"{icon} {t('画像アップロード')}"


def _input_mode_help() -> str:
    """FR-004: 選ぶ前に読める説明。

    `st.segmented_control` は選択肢ごとの `help` を持たないので、両モードを
    1 つの tooltip へまとめる。**定数ではなく関数にしてある**——module-level の
    定数にすると訳が import 時に固定される（FR-043）。
    """
    camera = t(
        "映像を見ながら、顔の向きや距離をその場で調整できます。"
        "数値は最大 8 Hz で更新され、映像は端末内のメモリだけで処理されます。"
    )
    image = t(
        "JPEG / PNG を 1 枚アップロードして分析します。"
        "10MB・12MP 以下、EXIF の向きを適用し、平滑化のない 1 回きりの結果を表示します。"
    )
    return f"**{t('Webカメラ')}**: {camera}\n\n**{t('画像アップロード')}**: {image}"


def _switch_input_mode() -> None:
    """FR-004: モードが変わったら、直前の入力の runtime 状態を破棄する。"""
    release_session_pipeline()
    release_image_landmarker()
    clear_session_result()


def _render_input_mode_selector() -> str:
    """FR-004: 入力方法を選ぶ。説明は `help` のツールチップが持つ（§5.3.1）。

    `st.segmented_control` は選択中の項目を解除でき、そのままではページに入力が
    1 つも無い状態になる。§15.1 は常にどれかが有効であることを求めるため、
    `None` を読んだときは空の画面を描かずに直前のモードへ戻す。

    **選択肢の語そのものが可視ラベルを兼ねる**（§5.4、2026-08-29）。
    「Webカメラ」「画像アップロード」は操作の名前そのものなので、同じ語を
    見出しでもう一度書かない。widget のラベルは `collapsed` にするが、
    Streamlit はそのラベルをアクセシブル名として保つ（AC-A11Y-01）。
    """
    # widget のキーではなく自前のキーから読む。widget が生成された後、その
    # キーは「解除」を意味する `None` を保持することがあり、それこそが
    # fallback が乗り越えなければならない値だからである。
    previous = st.session_state.get(ACTIVE_INPUT_MODE_KEY, INPUT_MODE_CAMERA)
    selected = st.segmented_control(
        t("入力方法"),
        options=list(_INPUT_MODES),
        format_func=_input_mode_label,
        key=INPUT_MODE_STATE_KEY,
        on_change=_switch_input_mode,
        help=_input_mode_help(),
        label_visibility="collapsed",
    )
    mode = selected if selected in _INPUT_MODES else previous
    st.session_state[ACTIVE_INPUT_MODE_KEY] = mode
    return mode


def _render_consent_gate() -> bool:
    """不変条件 3: 画像 upload と camera permission の前に明示同意を取る。

    すでに同意済みなら（直前の実行の widget 状態から判断して）畳んでおく。
    再訪した利用者が毎回この節をスクロールで通り過ぎずに済むようにする。
    """
    checked = [
        bool(st.session_state.get(key, False))
        for key in (CONSENT_PRIVACY_KEY, CONSENT_RESEARCH_KEY, CONSENT_RIGHTS_KEY)
    ]
    already_consented = all(checked)
    with st.expander(t("利用前に確認してください"), expanded=not already_consented):
        consent_privacy = st.checkbox(
            t("映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。"),
            key=CONSENT_PRIVACY_KEY,
        )
        consent_research = st.checkbox(
            t(
                "私は18〜75歳です。結果は参考値であり、実年齢、自己認識・法的性別、"
                "肌質・医学的状態、人物の価値を示さないことに同意します。"
            ),
            key=CONSENT_RESEARCH_KEY,
        )
        consent_rights = st.checkbox(
            t("本人の映像・画像、または分析する明示的な権限・同意があるものだけを使用します。"),
            key=CONSENT_RIGHTS_KEY,
        )
    return consent_privacy and consent_research and consent_rights


def render_app() -> None:
    """Analyze ページ: 同意 → 入力選択 → 分析（§5.1）。

    ページ最下部の概要区画は削除した（項目 31、H8、2026-08-29）。算出方法・
    制約・禁止用途は About ページが唯一の正本として持ち、About と参照統計へは
    サイドバーの multipage nav から到達する。
    """
    st.title("Face Symmetry Lab")
    st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")

    if not _render_consent_gate():
        st.info(t("3 項目を確認すると、カメラまたは画像を選択できます。"))
        return

    input_mode = _render_input_mode_selector()
    st.divider()

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        render_image_section()


def build_pages() -> list[StreamlitPage]:
    """Streamlit `st.navigation` へ渡すページ一式（§4.1、FR-043）。

    **module-level の定数にしない。** ナビの表示名は `t()` を通すので、
    module-level で組み立てると import した時点の言語で固定される。`ui.app` は
    entry script として毎回再実行されるため production では偶然動くが、
    module として import されたときだけ凍結する——同じ file が読まれ方によって
    2 通りに振る舞う状態を残さない。
    """
    return [
        st.Page(render_app, title="Analyze", icon="🪞", default=True),
        st.Page(render_reference_page, title=t("参照統計"), icon="📊"),
        st.Page(render_about_page, title=t("使い方・算出方法"), icon="📘"),
    ]


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    with st.sidebar:
        render_language_selector()
    st.navigation(build_pages()).run()
