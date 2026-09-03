"""FR-021: セッション要約の描画（`ui.summary_panel`）。

算術は `tests/domain/test_session_summary.py` が受け持つ。ここで見るのは
**要約が記述統計以上のことを主張しないこと**——順位化の語も価値判断語も出さず、
撮り直しで動く幅の注記を必ず伴うこと——と、欠測が `—` になることである。
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.summary_panel import (
    REFERENCE_AGE_RATE_LABEL,
    SUMMARY_HEADING,
    SUMMARY_LIMITS_NOTE,
    SUMMARY_SEGMENT_NOTE,
)

_FULL = (
    "from face_symmetry_lab.domain.models import ReasonCode, Status\n"
    "from face_symmetry_lab.domain.session_summary import "
    "SessionSummary, SummaryReading, SummarySection\n"
    "from face_symmetry_lab.ui.summary_panel import render_session_summary\n"
    "reading = SummaryReading(\n"
    "    median=98.12, minimum=96.05, maximum=99.20, iqr=0.83, sample_count=120\n"
    ")\n"
    "def section(segment_id, ok=120, total=150):\n"
    "    return SummarySection(\n"
    "        segment_id=segment_id,\n"
    "        total_count=total,\n"
    "        ok_count=ok,\n"
    "        valid_rate=ok / total,\n"
    "        valid_span_ms=42_000,\n"
    "        symmetry=reading,\n"
    "        parts={name: reading for name in ('contour', 'eyes', 'nose', 'mouth')},\n"
    "        reason_counts=((ReasonCode.OFF_AXIS_YAW, 20), (ReasonCode.TOO_FAR, 10)),\n"
    "        age_gate_evaluated=100,\n"
    "        age_gate_shown=62,\n"
    "    )\n"
    "render_session_summary(SessionSummary(\n"
    "    overall=section(None),\n"
    "    segments=(section('seg-1'), section('seg-2')),\n"
    "))\n"
)

_MISSING = (
    "from face_symmetry_lab.domain.session_summary import SessionSummary, SummarySection\n"
    "from face_symmetry_lab.ui.summary_panel import render_session_summary\n"
    "empty = SummarySection(\n"
    "    segment_id=None,\n"
    "    total_count=30,\n"
    "    ok_count=0,\n"
    "    valid_rate=0.0,\n"
    "    valid_span_ms=None,\n"
    "    symmetry=None,\n"
    "    parts={name: None for name in ('contour', 'eyes', 'nose', 'mouth')},\n"
    "    reason_counts=(),\n"
    ")\n"
    "render_session_summary(SessionSummary(overall=empty, segments=(empty,)))\n"
)


def _run(script: str = _FULL) -> AppTest:
    at = AppTest.from_string(script)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts: list[str] = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.subheader):
        parts.extend(str(element.value) for element in block)
    parts.extend(str(element.label) for element in at.metric)
    parts.extend(str(element.value) for element in at.metric)
    return " ".join(parts)


def test_the_summary_renders_without_raising() -> None:
    at = _run()

    assert not at.exception
    assert SUMMARY_HEADING in _all_text(at)


def test_the_valid_rate_is_shown_as_a_percentage() -> None:
    text = _all_text(_run())

    assert "80.0%" in text


def test_the_readings_are_shown_in_their_display_units() -> None:
    """§8.6.3: 対称度は %。"""
    text = _all_text(_run())

    assert "98.12%" in text


def test_every_displayed_reading_appears_in_the_table() -> None:
    text = _all_text(_run())

    for label in ("全体の対称度", "輪郭", "目", "鼻", "口"):
        assert label in text


def test_each_segment_is_labelled_and_traceable_to_the_exported_rows() -> None:
    """ADR-0009 (c): 区間が分かれていることと、その理由が表から読み取れる。

    `segment_id` の先頭 8 桁を添えるのは、エクスポートした CSV の `segment_id`
    列と突き合わせる唯一の手掛かりだからである（罠: 通し番号だけの実装だと、
    どの区間がどの行に対応するのか外から辿れない）。
    """
    text = _all_text(_run())

    assert "区間 1" in text
    assert "区間 2" in text
    assert "seg-1" in text
    assert "seg-2" in text
    assert SUMMARY_SEGMENT_NOTE in text


def test_the_top_hold_reasons_are_listed_with_counts() -> None:
    text = _all_text(_run())

    assert "20" in text
    assert "10" in text


def test_missing_readings_render_as_a_dash_rather_than_zero() -> None:
    """不変条件 11: OK 行が 0 件のセッションに 0.00% を並べない。"""
    at = _run(_MISSING)
    text = _all_text(at)

    assert not at.exception
    assert "—" in text
    assert "0.00%" not in text
    # 有効率だけは 0% であって欠測ではない（分母は確定している）。
    assert "0.0%" in text


def test_the_summary_states_that_it_moves_between_takes() -> None:
    """§14.4 の実測を注記として必ず添える。"""
    text = _all_text(_run())

    assert SUMMARY_LIMITS_NOTE in text
    assert "撮り直" in SUMMARY_LIMITS_NOTE


# --- §0 項目 34（撤回、2026-09-03）: 「全体のスコア」--------------------------


def test_the_summary_no_longer_shows_the_overall_score() -> None:
    """§8.6.7 の表示スケールを撤回した。読みは 5 つになり、「点」表記は現れない。"""
    text = _all_text(_run())

    assert "全体のスコア" not in text
    assert "点" not in text


@pytest.mark.parametrize(
    "banned",
    ["偏差値", "順位", "パーセンタイル", "合格", "合否", "平均より上", "平均より下"],
)
def test_the_summary_never_ranks_the_user(banned: str) -> None:
    """§10.7: 記述統計であって、順序への写像ではない。"""
    assert banned not in _all_text(_run())


@pytest.mark.parametrize("banned", ["美しい", "良い", "悪い", "優れ", "理想", "劣", "良好", "不良"])
def test_the_summary_carries_no_value_judgement(banned: str) -> None:
    assert banned not in _all_text(_run())


def test_the_summary_uses_no_progress_bar_or_gauge() -> None:
    """§5.2: 有効率を達成度として演出しない。"""
    import inspect

    from face_symmetry_lab.ui import summary_panel

    source = inspect.getsource(summary_panel)

    assert "st.progress" not in source
    assert "delta=" not in source, "上下の矢印は評価に読める（§5.2）"


def test_nothing_is_drawn_without_any_record() -> None:
    """記録が 1 件も無いセッションへ 0 を並べない。"""
    at = AppTest.from_string(
        "from face_symmetry_lab.domain.session_summary import SessionSummary\n"
        "from face_symmetry_lab.ui.summary_panel import render_session_summary\n"
        "render_session_summary(SessionSummary(overall=None, segments=()))\n"
    )
    at.run()

    assert not at.exception
    assert SUMMARY_HEADING not in _all_text(at)


# --- FR-021 (5): 参考年齢を表示できた割合（2026-08-16） -----------------------


def test_the_reference_age_rate_is_shown_when_the_gate_ran() -> None:
    text = _all_text(_run())

    assert REFERENCE_AGE_RATE_LABEL in text
    assert "62%" in text


def test_the_rate_is_absent_when_the_gate_never_ran() -> None:
    """0 件を 0% と書かない。測っていないことと「0% だった」は別である。"""
    text = _all_text(_run(_MISSING))

    assert REFERENCE_AGE_RATE_LABEL not in text


@pytest.mark.parametrize("banned", ["範囲外", "判定できません", "未成年", "18歳", "子ども"])
def test_the_rate_never_names_why_it_could_not_be_shown(banned: str) -> None:
    """§10.11-2: できなかった内訳を出さない。

    罠: 「範囲外だった割合」という裏返しのラベルは、per-frame では同一文言に
    して区別させなかったものを、セッション単位で復元できるようにする。
    """
    at = _run()
    text = _all_text(at) + " ".join(str(e.label) for e in at.metric)

    assert banned not in text
