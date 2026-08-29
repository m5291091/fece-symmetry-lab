"""§5.1: 映像・画像の下に並ぶタブの名前・順序・アイコン（`ui.result_tabs`）。

Camera と Image が別々に `st.tabs([...])` を書いていた間、両者のラベルは実際に
食い違っていた（片方は「まとめ・エクスポート」、もう片方は「エクスポート」）。
**この module の存在意義は、その食い違いが起こりえない形にすることである。**
そのためここでは「ラベルの並び」と「ラベルと中身の対応」の 2 つを固定する。

2026-08-29: 「左右対称との比較」は Image だけが持つタブへ変わり（項目 33、
FR-034）、「表示設定」はタブ列から外れて Camera の映像直下のポップオーバーへ
移った（項目 29）。この module から `settings` の概念そのものが消えたことも
ここで固定する。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.result_tabs import (
    DEMOGRAPHIC_TAB_LABEL,
    DETAIL_TAB_LABEL,
    REPORT_TAB_LABEL,
    SYMMETRY_TAB_LABEL,
    result_tab_labels,
)


def _tabs_app(*, with_symmetry: bool) -> AppTest:
    """`render_result_tabs` を 1 回だけ描画し、各タブに目印を書かせる。"""
    symmetry_line = (
        "        symmetry=lambda: st.markdown('body:symmetry'),\n" if with_symmetry else ""
    )
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.result_tabs import ResultTabs, render_result_tabs\n"
        "render_result_tabs(\n"
        "    ResultTabs(\n"
        "        detail=lambda: st.markdown('body:detail'),\n"
        "        demographic=lambda: st.markdown('body:demographic'),\n"
        "        report=lambda: st.markdown('body:report'),\n"
        f"{symmetry_line}"
        "    )\n"
        ")\n"
    )
    at.run()
    assert not at.exception
    return at


def test_the_shared_order_puts_the_symmetry_view_between_the_detail_and_the_reference_age() -> None:
    """§5.1（2026-08-29）: 共通は 詳細結果 → 参考年齢・参考性別 → レポート で、
    Image だけがその間に「左右対称との比較」を持つ。

    罠: 「左右対称との比較」を末尾へ足す実装は、入力を切り替えると「レポート」
    の位置が動く——同じものを探す先が動いてしまう。
    """
    assert result_tab_labels(with_symmetry=False) == (
        DETAIL_TAB_LABEL,
        DEMOGRAPHIC_TAB_LABEL,
        REPORT_TAB_LABEL,
    )
    assert result_tab_labels(with_symmetry=True) == (
        DETAIL_TAB_LABEL,
        SYMMETRY_TAB_LABEL,
        DEMOGRAPHIC_TAB_LABEL,
        REPORT_TAB_LABEL,
    )


def test_every_label_carries_its_own_body() -> None:
    """ラベルと中身の対応そのものを固定する。

    罠: ラベル列に「左右対称との比較」を差し込みながら、中身は 3 つのまま末尾へ
    追加する実装は、並びの test だけでは通ってしまう。「左右対称との比較」タブを
    開くとレポートが出る、という取り違えはここでしか捕まらない。
    """
    for with_symmetry in (False, True):
        at = _tabs_app(with_symmetry=with_symmetry)
        rendered = {tab.label: [element.value for element in tab.markdown] for tab in at.tabs}

        expected = {
            DETAIL_TAB_LABEL: ["body:detail"],
            DEMOGRAPHIC_TAB_LABEL: ["body:demographic"],
            REPORT_TAB_LABEL: ["body:report"],
        }
        if with_symmetry:
            expected[SYMMETRY_TAB_LABEL] = ["body:symmetry"]

        assert rendered == expected, with_symmetry


def test_the_camera_mode_gets_exactly_one_tab_fewer() -> None:
    """Camera は左右対称との比較を持たない（§0 項目 33）。それ以外は 1 つも減らさない。"""
    without = _tabs_app(with_symmetry=False)
    with_symmetry = _tabs_app(with_symmetry=True)

    assert [tab.label for tab in without.tabs] == list(result_tab_labels(with_symmetry=False))
    assert [tab.label for tab in with_symmetry.tabs] == list(result_tab_labels(with_symmetry=True))


def test_the_icons_never_stand_in_for_the_wording() -> None:
    """§5.4: アイコンは見分けの手掛かりであって、文言の置き換えではない。

    §5.2: 達成度・良し悪しを表す絵文字は使わない。
    """
    wordings = {
        DETAIL_TAB_LABEL: "詳細結果",
        SYMMETRY_TAB_LABEL: "左右対称との比較",
        DEMOGRAPHIC_TAB_LABEL: "参考年齢・参考性別",
        REPORT_TAB_LABEL: "レポート",
    }
    for label, wording in wordings.items():
        assert wording in label
        # アイコン 1 文字ぶんだけ長い(文言を絵文字へ置き換えていない)。
        assert label != wording

    banned = ("✅", "❌", "⭕", "❤️", "👍", "👎", "🏆", "⭐", "🎉")
    for label in wordings:
        for emoji in banned:
            assert emoji not in label, label


def test_the_settings_tab_no_longer_exists() -> None:
    """§0 項目 29（2026-08-29）: 表示設定はタブではなく映像直下のポップオーバーへ
    移った。この module から `settings` の概念そのものが無いことを固定する。

    罠: 定数だけ残して誰も使わない実装は、次に触る人が復活させる。
    """
    import inspect
    from dataclasses import fields

    from face_symmetry_lab.ui import result_tabs

    assert not hasattr(result_tabs, "SETTINGS_TAB_LABEL")
    assert "settings" not in {field.name for field in fields(result_tabs.ResultTabs)}
    assert "with_settings" not in inspect.signature(result_tabs.result_tab_labels).parameters
