"""§5.1: 映像・画像の下に並ぶタブの名前・順序・アイコン（`ui.result_tabs`）。

Camera と Image が別々に `st.tabs([...])` を書いていた間、両者のラベルは実際に
食い違っていた（片方は「まとめ・エクスポート」、もう片方は「エクスポート」）。
**この module の存在意義は、その食い違いが起こりえない形にすることである。**
そのためここでは「ラベルの並び」と「ラベルと中身の対応」の 2 つを固定する。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.result_tabs import (
    DEMOGRAPHIC_TAB_LABEL,
    DETAIL_TAB_LABEL,
    REPORT_TAB_LABEL,
    SETTINGS_TAB_LABEL,
    SYMMETRY_TAB_LABEL,
    result_tab_labels,
)


def _tabs_app(*, with_settings: bool) -> AppTest:
    """`render_result_tabs` を 1 回だけ描画し、各タブに目印を書かせる。"""
    settings_line = "    settings=lambda: st.markdown('body:settings'),\n" if with_settings else ""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.result_tabs import ResultTabs, render_result_tabs\n"
        "render_result_tabs(\n"
        "    ResultTabs(\n"
        "        detail=lambda: st.markdown('body:detail'),\n"
        "        symmetry=lambda: st.markdown('body:symmetry'),\n"
        "        demographic=lambda: st.markdown('body:demographic'),\n"
        "        report=lambda: st.markdown('body:report'),\n"
        f"    {settings_line}"
        "    )\n"
        ")\n"
    )
    at.run()
    assert not at.exception
    return at


def test_the_shared_order_puts_settings_between_the_reference_age_and_the_report() -> None:
    """§5.1（2026-08-26）: 共通は 詳細結果 → 左右対称との比較 → 参考年齢・参考性別
    → レポート で、Camera だけがその間に「表示設定」を持つ。

    罠: 「表示設定」を末尾へ足す実装は、Camera と Image でレポートの位置が
    ずれる——入力を切り替えたときに同じものを探す先が動く。
    """
    assert result_tab_labels(with_settings=False) == (
        DETAIL_TAB_LABEL,
        SYMMETRY_TAB_LABEL,
        DEMOGRAPHIC_TAB_LABEL,
        REPORT_TAB_LABEL,
    )
    assert result_tab_labels(with_settings=True) == (
        DETAIL_TAB_LABEL,
        SYMMETRY_TAB_LABEL,
        DEMOGRAPHIC_TAB_LABEL,
        SETTINGS_TAB_LABEL,
        REPORT_TAB_LABEL,
    )


def test_every_label_carries_its_own_body() -> None:
    """ラベルと中身の対応そのものを固定する。

    罠: ラベル列に「表示設定」を差し込みながら、中身は 4 つのまま末尾へ追加
    する実装は、並びの test だけでは通ってしまう。「表示設定」タブを開くと
    レポートが出る、という取り違えはここでしか捕まらない。
    """
    for with_settings in (False, True):
        at = _tabs_app(with_settings=with_settings)
        rendered = {tab.label: [element.value for element in tab.markdown] for tab in at.tabs}

        expected = {
            DETAIL_TAB_LABEL: ["body:detail"],
            SYMMETRY_TAB_LABEL: ["body:symmetry"],
            DEMOGRAPHIC_TAB_LABEL: ["body:demographic"],
            REPORT_TAB_LABEL: ["body:report"],
        }
        if with_settings:
            expected[SETTINGS_TAB_LABEL] = ["body:settings"]

        assert rendered == expected, with_settings


def test_the_image_mode_gets_exactly_one_tab_fewer() -> None:
    """Image は表示設定を持たない（§5.1）。それ以外は 1 つも減らさない。"""
    without = _tabs_app(with_settings=False)
    with_settings = _tabs_app(with_settings=True)

    assert [tab.label for tab in without.tabs] == list(result_tab_labels(with_settings=False))
    assert [tab.label for tab in with_settings.tabs] == list(result_tab_labels(with_settings=True))


def test_the_icons_never_stand_in_for_the_wording() -> None:
    """§5.4: アイコンは見分けの手掛かりであって、文言の置き換えではない。

    §5.2: 達成度・良し悪しを表す絵文字は使わない。
    """
    wordings = {
        DETAIL_TAB_LABEL: "詳細結果",
        SYMMETRY_TAB_LABEL: "左右対称との比較",
        DEMOGRAPHIC_TAB_LABEL: "参考年齢・参考性別",
        SETTINGS_TAB_LABEL: "表示設定",
        REPORT_TAB_LABEL: "レポート",
    }
    for label, wording in wordings.items():
        assert wording in label
        # アイコン 1 文字ぶんだけ長い（文言を絵文字へ置き換えていない）。
        assert label != wording

    banned = ("✅", "❌", "⭕", "❤️", "👍", "👎", "🏆", "⭐", "🎉")
    for label in wordings:
        for emoji in banned:
            assert emoji not in label, label
