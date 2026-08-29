"""UI 表示言語の切替（FR-043）。

**日本語の原文をそのまま鍵にする**（gettext の msgid と同じ作法、ADR-0010 (a)）。
記号鍵にすると、`ui/*.py` の文言リテラルが別モジュールへ移ることになり、
`tests/ui/` にある「ソース文字列に禁止語が無いこと」を `inspect.getsource` で
確かめる構造テスト群が、その瞬間からすべて空振りで通るようになる。原文を鍵に
して呼び出し側のリテラルを動かさなければ、それらは今までどおり働き続ける。

**訳が無い鍵は日本語の原文を返す**（fail-open、ADR-0010 (b)）。英語表示中に
空文字や鍵名が出るより日本語が残るほうが害が小さい。訳し忘れは
`tests/ui/test_i18n.py` が AST 走査で機械的に落とすので、fail-open が
抜けを隠す経路にはならない。
"""

from __future__ import annotations

from typing import Final, Literal, get_args

import streamlit as st

from face_symmetry_lab.ui.translations import EN

Language = Literal["ja", "en"]

LANGUAGE_STATE_KEY: Final[str] = "ui_language"
DEFAULT_LANGUAGE: Final[Language] = "ja"
LANGUAGES: Final[tuple[Language, ...]] = get_args(Language)
#: 言語名は**その言語自身で書く**。英語しか読めない利用者が「日本語」を
#: 探せるようにするためである（`Japanese` と書くと日本語話者が探しにくい）。
LANGUAGE_LABELS: Final[dict[Language, str]] = {"ja": "日本語", "en": "English"}
LANGUAGE_SELECT_LABEL: Final[str] = "Language / 言語"


def current_language() -> Language:
    """いま有効な表示言語。未設定・不正値・スクリプト文脈の外では既定へ落ちる。

    `st.session_state` は Streamlit のスクリプト実行文脈の外では例外を投げる。
    `ui.app` は module-level で `st.Page` を組み立てており、テストが素の import
    で読むことがあるため、ここで握って既定を返す（不変条件 12 の失敗分離）。
    """
    try:
        value = st.session_state.get(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
    except Exception:
        return DEFAULT_LANGUAGE
    return value if value in LANGUAGES else DEFAULT_LANGUAGE


def t(text: str) -> str:
    """表示言語に合わせて訳す。訳が無ければ原文をそのまま返す。"""
    if current_language() == "en":
        return EN.get(text, text)
    return text


def translate_message(text: str) -> str:
    """実行時にしか値が決まらない文字列（例外メッセージ）専用の訳出。

    `t()` と同じ辞書引きだが、名前が `t` ではないため `tests/ui/test_i18n.py`
    の AST 規約（`t(...)` の引数はリテラルか module-level の名前・辞書添字で
    なければならない）の対象にならない。**画面へ直接書く文言には使わない**
    ——静的な文言は必ず `t("...")` を使うこと。

    `domain` / `application` / `adapters` の例外は文言を持つが `t()` を呼べない
    （CLAUDE.md §6: adapters に表示文言を持たせない）。`ui` 側の境界で
    `translate_message(str(error))` として訳す（FR-043）。値を埋め込んで組み立てた
    f-string の例外文言は原文と一致しないため鍵として引けず、fail-open のまま
    日本語で残る（許容範囲、`translations/_errors.py`）。
    """
    if current_language() == "en":
        return EN.get(text, text)
    return text


def render_language_selector() -> Language:
    """全ページ共通の言語選択（§4.1、§5.4）。サイドバーへ 1 つだけ置く。

    **既知の制限**: `ui/hud_assets.py` は擬似全画面中に `.stSidebar` を
    `display:none` にするため、カメラ全画面中はこの selector に到達できず
    言語を切り替えられない。全画面を抜ければ戻る。
    """
    st.radio(
        LANGUAGE_SELECT_LABEL,
        options=LANGUAGES,
        format_func=lambda code: LANGUAGE_LABELS[code],
        key=LANGUAGE_STATE_KEY,
        horizontal=True,
    )
    return current_language()
