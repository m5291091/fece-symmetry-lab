"""FR-022/FR-024/§5.1: 参考年齢・参考性別の独立区画（`ui.demographic_panel`）。

Camera と Image がそれぞれ自前で書いていた区画を 1 つにしたもの。**分けていた
間、両者は状態の並べ方が違っていた**——Camera は「モデル未配置」を最初に見て、
Image は最後のフォールバックとして見ていたため、モデルが無い写真では「単一顔を
検出できた写真でのみ…」という無関係な理由が出ていた。ここでの test は
「どの状態でも黙らない」ことと「ゲートの判断がどの入力でも同じに効く」ことを
1 か所で押さえる。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus
from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.ui.age_gate_notice import age_gate_notice
from face_symmetry_lab.ui.demographic_panel import (
    DISPLAY_OFF_NOTICE,
    MODEL_MISSING_NOTICE,
)
from face_symmetry_lab.ui.scores import DEMOGRAPHIC_RUNTIME_CAPTION

_PENDING = "まだ推定できていません。"


def _demographic_estimate(age: float = 31.0) -> DemographicEstimate:
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(0.2, 0.8),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 27, tzinfo=UTC),
    )


def _gate(status: AgeGateStatus) -> AgeGateOutcome:
    return AgeGateOutcome(status=status, decision=None, reason=None)


def _panel(**overrides: Any) -> AppTest:
    """共有パネルを 1 状態だけ描画する。"""
    settings: dict[str, Any] = {
        "estimate": _demographic_estimate(),
        "gate_outcome": None,
        "available": True,
        "show_demographics": True,
        "pending_notice": _PENDING,
        "error": None,
        "footnote": None,
    }
    settings.update(overrides)
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.demographic_panel import render_demographic_panel\n"
        "render_demographic_panel(**st.session_state['args'])\n"
    )
    at.session_state["args"] = settings
    at.run()
    assert not at.exception
    return at


def test_the_panel_shows_the_age_and_the_gender_together() -> None:
    """依頼（2026-08-27）: どちらの入力でも参考年齢と参考性別を同じ形で出す。"""
    at = _panel()

    assert [metric.value for metric in at.metric] == ["31.0歳", "男性"]


def test_the_panel_obeys_the_gate_like_every_other_area() -> None:
    """§10.12-3: HUD・吹き出しが隠した判断を、タブから読めるようにしてはならない。

    罠: `estimate is not None` だけで描く実装は、HUD が固定文言へ切り替わって
    いる間もタブに参考年齢を出し続ける。
    """
    at = _panel(gate_outcome=_gate(AgeGateStatus.OUT_OF_RANGE))

    assert not at.metric
    assert age_gate_notice(AgeGateStatus.OUT_OF_RANGE) in [info.value for info in at.info]


def test_the_panel_never_goes_silent() -> None:
    """FR-024/§5.1: 何も出せないときも、何が起きているかを必ず 1 行で述べる。

    罠: 推定が無いときに `return` するだけの実装は、利用者から見て「壊れている」
    と「まだ推定できていない」を区別できない状態になる。
    """
    silent: tuple[dict[str, Any], ...] = (
        {"available": False},
        {"show_demographics": False},
        {"estimate": None},
        {"estimate": None, "error": "推定できませんでした。"},
        {"gate_outcome": _gate(AgeGateStatus.NOT_EVALUATED)},
        {"estimate": None, "gate_outcome": _gate(AgeGateStatus.UNDETERMINED)},
    )
    for overrides in silent:
        at = _panel(**overrides)

        assert at.info or at.warning or at.error, overrides


def test_a_missing_model_is_reported_before_anything_else() -> None:
    """モデルが無いときに「単一顔を検出できた写真でのみ…」と言わない。

    罠: モデル未配置を最後のフォールバックに置く実装（Image の旧実装）は、
    利用者へ手の打ちようがない理由を示す。
    """
    at = _panel(available=False, estimate=None, pending_notice=_PENDING)

    infos = [info.value for info in at.info]
    assert MODEL_MISSING_NOTICE in infos
    assert _PENDING not in infos
    assert not at.metric


def test_turning_the_display_off_is_not_reported_as_a_failure() -> None:
    """FR-024: 利用者が切っただけの状態を「表示できません」と説明しない。"""
    at = _panel(show_demographics=False, gate_outcome=_gate(AgeGateStatus.OUT_OF_RANGE))

    infos = [info.value for info in at.info]
    assert DISPLAY_OFF_NOTICE in infos
    assert age_gate_notice(AgeGateStatus.OUT_OF_RANGE) not in infos
    assert not at.metric


def test_a_failed_estimate_states_its_reason_instead_of_the_pending_line() -> None:
    """不変条件 12: 失敗はこの区画だけに現れ、しかも黙って消えない。"""
    at = _panel(estimate=None, error="年齢・性別を推定できませんでした。")

    assert "年齢・性別を推定できませんでした。" in [error.value for error in at.error]
    assert _PENDING not in [info.value for info in at.info]


def test_the_gate_notice_wins_over_the_error_line() -> None:
    """§10.11-2: ゲートが隠した状態で、推論の失敗理由を代わりに見せない。

    罠: `error` を先に見る実装は、ゲートが働いた回と推論が落ちた回を利用者から
    区別できるようにしてしまう。
    """
    at = _panel(
        estimate=None,
        error="年齢・性別を推定できませんでした。",
        gate_outcome=_gate(AgeGateStatus.OUT_OF_RANGE),
    )

    assert not at.error
    assert age_gate_notice(AgeGateStatus.OUT_OF_RANGE) in [info.value for info in at.info]


def test_the_provenance_line_is_always_present() -> None:
    """§10.8: 来歴の 1 行は、出せた状態でも出せない状態でも同じように添える。"""
    for overrides in ({}, {"available": False}, {"estimate": None}):
        at = _panel(**overrides)

        assert DEMOGRAPHIC_RUNTIME_CAPTION in [caption.value for caption in at.caption], overrides


def test_the_footnote_is_appended_only_when_the_caller_has_one() -> None:
    """§9.6.1: Camera はセッション範囲を添え、Image は添えるものを持たない。"""
    with_footnote = _panel(footnote="このセッションで表示できた参考年齢 31 歳")
    without = _panel()

    captions = [caption.value for caption in with_footnote.caption]
    assert "このセッションで表示できた参考年齢 31 歳" in captions
    assert len(captions) > len([caption.value for caption in without.caption])
