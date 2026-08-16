"""FR-040: 表示平滑化時間を選ぶ UI（`ui.smoothing_panel`）。

この panel が守るのは 3 点である。

1. **選べるのは §8.7 の時定数だけで、測定値の作り方は選べない。** 文言がここを
   曖昧にすると、利用者は「数値そのものを調整している」と読む。
2. **選択の意味を数値で開示する。** 追従時間と実効サンプル数はいずれも §8.7 の
   式から導かれるだけの量で、新しい定数を持たない（ADR-0007 (c)）。
3. **推奨値・到達目標を置かない**（§5.3）。「おすすめ」「最適」は、この設定に
   正解があるという主張になる。
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.smoothing import (
    DEFAULT_TAU_SECONDS,
    TAU_CHOICES_SECONDS,
    effective_sample_count,
    response_time_seconds,
)
from face_symmetry_lab.ui.smoothing_panel import (
    HOLD_EFFECT_NOTE,
    HOLD_MISSING_NOTE,
    HOLD_STATE_KEY,
    SMOOTHING_RESET_NOTE,
    SMOOTHING_SCOPE_NOTE,
    SMOOTHING_STATE_KEY,
    SMOOTHING_TRADEOFF_NOTE,
    nearest_hold_choice,
)


def _run(
    tau: float | None = DEFAULT_TAU_SECONDS,
    interval_ms: int = 125,
    hold: float | None = None,
) -> AppTest:
    at = AppTest.from_string(
        "from face_symmetry_lab.ui.smoothing_panel import render_smoothing_control\n"
        "render_smoothing_control(\n"
        f"    tau_seconds={tau!r}, analysis_interval_ms={interval_ms}, hold_seconds={hold!r}\n"
        ")\n"
    )
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts: list[str] = []
    for block in (at.markdown, at.caption, at.info, at.warning):
        parts.extend(str(element.value) for element in block)
    parts.extend(str(element.label) for element in at.metric)
    parts.extend(str(element.value) for element in at.metric)
    return " ".join(parts)


def test_the_control_renders_one_step_selector_per_documented_ladder() -> None:
    """FR-040 の「量」と FR-041 の「頻度」で 2 つ。どちらも段から選ぶ。"""
    at = _run()

    assert not at.exception
    assert len(at.select_slider) == 2
    assert tuple(at.select_slider[0].options) == tuple(
        f"{tau:.1f} 秒" for tau in TAU_CHOICES_SECONDS
    )
    assert tuple(at.select_slider[1].options) == ("毎回", "0.5 秒", "1.0 秒", "2.0 秒", "5.0 秒")


def test_the_selector_starts_at_the_value_the_pipeline_reports() -> None:
    """罠: 既定を決め打ちする実装は、別の値で動いている pipeline と食い違う。"""
    at = _run(tau=2.0)

    assert at.session_state[SMOOTHING_STATE_KEY] == "2.0 秒"


def test_a_time_constant_off_the_ladder_snaps_to_the_nearest_step() -> None:
    """`st.select_slider` は options に無い値を渡すと例外になる。

    段の外の `tau`（過去の設定・将来の既定変更）で画面が落ちてはならない。
    """
    at = _run(tau=0.45)

    assert not at.exception
    assert at.session_state[SMOOTHING_STATE_KEY] == "0.4 秒"


def test_the_control_survives_a_pipeline_without_smoothing() -> None:
    at = _run(tau=None)

    assert not at.exception
    assert at.session_state[SMOOTHING_STATE_KEY] == f"{DEFAULT_TAU_SECONDS:.1f} 秒"


def test_render_returns_both_selected_settings() -> None:
    """呼び出し側（`ui.camera`）が pipeline へ適用できるよう、両方を返す。"""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.smoothing_panel import render_smoothing_control\n"
        "st.session_state['selected'] = render_smoothing_control(\n"
        "    tau_seconds=1.0, analysis_interval_ms=125, hold_seconds=2.0\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.session_state["selected"].tau_seconds == pytest.approx(1.0)
    assert at.session_state["selected"].hold_seconds == pytest.approx(2.0)


def test_the_readouts_come_from_the_same_formulas_as_the_filter() -> None:
    """ADR-0007 (c): 開示する量は §8.7 の式から導かれるだけで、新しい定数を持たない。

    罠: 95% 追従を `3 × tau` と近似した実装、実効サンプル数を解析周期と無関係に
    出す実装は、いずれもここで落ちる。
    """
    at = _run(tau=1.0, interval_ms=200)
    text = _all_text(at)

    assert f"{response_time_seconds(1.0, fraction=0.95):.1f} 秒" in text
    assert f"{effective_sample_count(1.0, interval_ms=200):.1f} 個" in text
    # 63% 追従は tau そのもの。式の意味が読み取れるよう、割合も画面に出す。
    assert "63" in text
    assert "95" in text


def test_the_readouts_follow_the_analysis_cadence() -> None:
    """罠: 解析周期を無視した実装は、どの cadence でも同じ本数を出す。"""
    fast = _all_text(_run(tau=1.0, interval_ms=125))
    slow = _all_text(_run(tau=1.0, interval_ms=333))

    assert f"{effective_sample_count(1.0, interval_ms=125):.1f} 個" in fast
    assert f"{effective_sample_count(1.0, interval_ms=333):.1f} 個" in slow
    assert fast != slow


def test_the_copy_states_what_the_setting_does_not_change() -> None:
    """FR-040: 表示の時間応答だけを変え、指標・生値・エクスポートを変えない。"""
    text = _all_text(_run())

    assert SMOOTHING_SCOPE_NOTE in text
    assert "計算式" in SMOOTHING_SCOPE_NOTE
    assert "生値" in SMOOTHING_SCOPE_NOTE


def test_the_copy_states_the_trade_off_in_both_directions() -> None:
    text = _all_text(_run())

    assert SMOOTHING_TRADEOFF_NOTE in text
    assert "短く" in SMOOTHING_TRADEOFF_NOTE
    assert "長く" in SMOOTHING_TRADEOFF_NOTE


def test_the_copy_states_that_changing_it_restarts_the_average() -> None:
    """§8.8: EMA リセットと新しい区間は、利用者から見える挙動である。"""
    text = _all_text(_run())

    assert SMOOTHING_RESET_NOTE in text
    assert "区間" in SMOOTHING_RESET_NOTE


@pytest.mark.parametrize(
    "banned",
    ["おすすめ", "推奨", "最適", "正しい値", "美しい", "良い", "悪い", "優れ", "理想", "劣"],
)
def test_the_panel_never_points_at_a_target_value(banned: str) -> None:
    """§5.3: この設定に「正解」は無い。示すのは交換関係だけである。"""
    at = _run()
    text = _all_text(at)
    labels = " ".join(str(element.label) for element in at.select_slider)

    assert banned not in text
    assert banned not in labels


def test_the_panel_uses_no_progress_bar_or_gauge() -> None:
    """§5.2: 設定値を達成度として演出しない。"""
    import inspect

    from face_symmetry_lab.ui import smoothing_panel

    source = inspect.getsource(smoothing_panel)

    assert "st.progress" not in source
    assert "st.slider(" not in source, "連続スライダーは ADR-0007 (c) で採らない"


# --- FR-041: 数値を入れ替える間隔 ---------------------------------------------


def test_the_hold_selector_starts_at_every_frame_by_default() -> None:
    """既定は「毎回」。この機能を知らない利用者の画面は変わらない。"""
    at = _run()

    assert at.session_state[HOLD_STATE_KEY] == "毎回"


def test_the_hold_selector_starts_at_the_value_the_pipeline_reports() -> None:
    at = _run(hold=2.0)

    assert at.session_state[HOLD_STATE_KEY] == "2.0 秒"


def test_a_hold_off_the_ladder_snaps_to_the_nearest_step() -> None:
    at = _run(hold=1.9)

    assert not at.exception
    assert at.session_state[HOLD_STATE_KEY] == "2.0 秒"


def test_zero_seconds_reads_as_a_word_rather_than_a_duration() -> None:
    """「0.0 秒ごとに更新」は意味が取れない。"""
    assert nearest_hold_choice(0.0) == pytest.approx(0.0)
    at = _run(hold=0.0)

    assert "毎回" in tuple(at.select_slider[1].options)
    assert at.session_state[HOLD_STATE_KEY] == "毎回"


def test_the_copy_explains_why_a_longer_average_is_not_enough() -> None:
    """この 1 文が無いと、2 つの設定が同じことをする 2 通りのやり方に見える。"""
    text = _all_text(_run())

    assert HOLD_EFFECT_NOTE in text
    assert "動く回数は変わりません" in HOLD_EFFECT_NOTE


def test_the_copy_promises_that_a_stale_number_never_survives_a_lost_face() -> None:
    """AC-HOLD-02 は実装の約束であり、画面にも書く（止まった数値の有効性）。"""
    text = _all_text(_run())

    assert HOLD_MISSING_NOTE in text
    assert "—" in HOLD_MISSING_NOTE


def test_no_heading_repeats_a_widget_label() -> None:
    """見出しと widget ラベルが同じだと、同じ文字列が 2 行続くだけになる。

    実ブラウザで 2 度見つけた回帰なので、ここで固定する。1 度目は呼び出し側
    （`ui.camera`）と `SMOOTHING_HEADING` の重複、2 度目はこの panel の中の
    小見出しと更新間隔のラベルの重複だった。
    """
    at = _run()

    headings = {str(element.value).lstrip("# ").strip() for element in at.markdown}
    labels = {str(element.label).strip() for element in at.select_slider}

    assert headings & labels == set()
