"""§5.1: 「詳細結果」タブの枠（`ui.detail_panel`）。

数値そのものの整形は `test_scores.py` が受け持つ。ここで見るのは **Camera と
Image で同じ形になっているか**——囲み・空のときの説明・診断表示の置き場所——
だけである。分けていた間、Image は入力サイズとランドマーク点数を測定結果より
**上**に置き、Camera は同じ性質の情報を折りたたみの**下**に置いていた。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.detail_panel import DIAGNOSTICS_LABEL
from face_symmetry_lab.ui.scores import SCORE_HEADING

_EMPTY = "この画像からは左右差を算出できませんでした。"

_OUTCOME = (
    "_outcome_for_display(\n"
    "    symmetry_3d=_symmetry_3d_result(\n"
    "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
    "    ),\n"
    "    smooth_3d=0.018,\n"
    ")"
)


def _panel(*, outcome: str, extra: str = "") -> AppTest:
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.detail_panel import render_detail_panel\n"
        f"render_detail_panel(\n"
        f"    {outcome},\n"
        f'    empty_notice="{_EMPTY}",\n'
        f"    show_parts=True,\n"
        f"{extra}"
        f")\n"
    )
    at.run()
    assert not at.exception
    return at


def test_the_panel_explains_itself_when_there_is_nothing_to_show() -> None:
    """FR-024: 空の枠を出さない。罠: `return` するだけの実装は真っ白になる。"""
    at = _panel(outcome="None")

    assert _EMPTY in [info.value for info in at.info]
    assert SCORE_HEADING not in [element.value for element in at.subheader]


def test_the_panel_draws_the_score_details_when_there_is_an_outcome() -> None:
    at = _panel(outcome=_OUTCOME)

    assert SCORE_HEADING in [element.value for element in at.subheader]
    assert not at.info


def test_the_part_breakdown_follows_the_flag_the_caller_passes() -> None:
    """AC-LAYOUT-01: Camera は HUD が部位別を出すため詳細タブでは省き、Image は残す。

    罠: 枠だけを共有して `show_parts` を握りつぶす実装は、Camera の HUD と
    詳細タブに同じ部位別を二重に出す。
    """
    shown = _panel(outcome=_OUTCOME)
    hidden = AppTest.from_string(
        "import streamlit as st\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.detail_panel import render_detail_panel\n"
        f"render_detail_panel({_OUTCOME}, empty_notice='x', show_parts=False)\n"
    )
    hidden.run()
    assert not hidden.exception

    for part in ("輪郭", "目", "鼻", "口"):
        assert part in [metric.label for metric in shown.metric], part
        assert part not in [metric.label for metric in hidden.metric], part


def test_the_diagnostics_live_behind_the_same_expander_in_both_modes() -> None:
    """§14.3f: 利用者が手を打てない実行時の数値は、結果の下の折りたたみへ入れる。

    罠: 呼び出し側が自前で `st.expander` を書く実装は、モードごとに見出しも
    位置も変わる（Image は測定結果より上に素の caption を置いていた）。
    """
    at = _panel(outcome=_OUTCOME, extra="    diagnostics=lambda: st.caption('diag:here'),\n")

    assert DIAGNOSTICS_LABEL in [expander.label for expander in at.expander]
    assert "diag:here" in [caption.value for caption in at.caption]


def test_the_panel_stays_quiet_when_the_caller_has_no_diagnostics() -> None:
    """Image は診断を持たない状態もありうる。空の折りたたみを置かない。"""
    at = _panel(outcome=_OUTCOME)

    assert DIAGNOSTICS_LABEL not in [expander.label for expander in at.expander]


def test_the_recovery_warning_comes_before_the_numbers() -> None:
    """§14.4: 回復導線は、読み飛ばされる位置へ置かない。"""
    at = _panel(outcome=_OUTCOME, extra="    warning='処理が続けて失敗しています。',\n")

    assert "処理が続けて失敗しています。" in [warning.value for warning in at.warning]
