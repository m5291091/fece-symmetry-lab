"""`ui.readings`: 表示値をどこから読み、どう整形するか。

`ui.scores`（描画）から分離した層に対応する。ここで守るのは 3 点である。

1. **同じ測定値に 2 つ目の表示規則を作らない。** HUD・吹き出し・詳細パネルは
   すべてこの module を通る。
2. **主指標は 3D である**（§8.6.3、項目 22）。fixture は 2D 側へ大きく離れた値
   （0.5）を入れてあるので、誤って 2D を読む実装はここで落ちる。
3. **欠測は `—`。** 0 でも前回値でもない（不変条件 11、NFR-013）。
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.readings import (
    GATE_COLORS,
    display_signature,
    format_display_hold,
    format_distance,
    format_smoothing_window,
    format_symmetry,
    gate_color,
    gate_guidance,
    gate_label,
    gate_warning,
    resolve_overall_distance,
    resolve_part_distance_3d,
    symmetry_percent,
)
from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result

# --- §8.6.3 対称度パーセントの表示 ------------------------------------------


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, "100.00%"),
        (0.0215, "97.85%"),
        (0.5, "50.00%"),
        (1.0, "0.00%"),
    ],
)
def test_a_distance_is_shown_as_its_complement(distance: float, expected: str) -> None:
    """§8.6.3: `100% - ずれ%`, with no coefficient anywhere in the transform."""
    assert format_symmetry(distance) == expected


def test_the_symmetry_and_deviation_readings_always_sum_to_one_hundred() -> None:
    """AC-DISPLAY-02: 2 つの向きは同じ測定値である。"""
    for distance in (0.0, 0.004, 0.0215, 0.31, 0.999):
        symmetry = float(format_symmetry(distance).rstrip("%"))
        deviation = float(format_distance(distance).rstrip("%"))
        assert symmetry + deviation == pytest.approx(100.0, abs=0.02)


def test_a_deviation_beyond_the_interocular_distance_floors_at_zero() -> None:
    """§8.6.3 がクランプするのは *表示* だけで、`d` 自体は上限を持たない。"""
    assert format_symmetry(4.0) == "0.00%"
    assert format_distance(4.0) == "400.00%"


def test_a_missing_distance_has_no_symmetry_reading() -> None:
    """NFR-013: 欠測が「完全に対称な 100%」と読めてはならない。"""
    assert format_symmetry(None) == "—"


def test_symmetry_percent_agrees_with_the_formatted_string() -> None:
    """H1: HUD の折れ線グラフはテキストと同じ変換を数値のまま使う。

    罠: グラフ側だけ別の式（生の距離、別の clamp）を持つ実装は、同じ画面で
    テキストと線が食い違う。
    """
    for distance in (0.0, 0.0215, 0.31, 1.0, 4.0):
        percent = symmetry_percent(distance)
        assert percent is not None
        assert f"{percent:.2f}%" == format_symmetry(distance)


def test_symmetry_percent_of_a_missing_value_is_missing() -> None:
    assert symmetry_percent(None) is None


# --- §8.6.3/§8.6.5: 表示・エクスポートを 3D 指標へ切り替える -----------------


def test_the_resolved_overall_distance_prefers_the_three_dimensional_smoothed_value() -> None:
    """§8.6.3/§8.7: 主表示は 3D の平滑値であり、2D の値ではない（罠 (a)）。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.0172,
        symmetry_2d_overall=0.5,
    )

    assert resolve_overall_distance(outcome) == pytest.approx(0.0172)


def test_the_resolved_overall_distance_falls_back_to_the_raw_three_dimensional_value() -> None:
    """§8.7: 静止画は平滑値を持たない。3D の生値へ落ち、2D へは落ちない。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=None,
        symmetry_2d_overall=0.5,
    )

    assert resolve_overall_distance(outcome) == pytest.approx(0.018)


def test_the_resolved_overall_distance_is_missing_without_a_three_dimensional_result() -> None:
    """不変条件 11: 3D 一式が欠測なら `—`。2D の値で埋めない。"""
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None, symmetry_2d_overall=0.5)

    assert resolve_overall_distance(outcome) is None


def test_resolve_part_distance_prefers_the_smoothed_value_over_the_raw_one() -> None:
    """§8.7 改訂（H1）: 部位別も全体値と同じく平滑値を主表示にする。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.011, eyes=0.022, nose=0.033, mouth=0.024, overall=0.018
        ),
        smooth_3d=0.018,
        part_smooth_3d={"contour": 0.014, "eyes": 0.016, "nose": 0.019, "mouth": 0.015},
    )

    assert resolve_part_distance_3d(outcome, "contour") == pytest.approx(0.014)
    assert resolve_part_distance_3d(outcome, "nose") == pytest.approx(0.019)


def test_resolve_part_distance_falls_back_to_raw_when_the_smoothed_container_is_missing() -> None:
    """静止画（§8.7）は部位別平滑値を持たない。生値へ自然に落ちる。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.011, eyes=0.022, nose=0.033, mouth=0.024, overall=0.018
        ),
        smooth_3d=None,
        part_smooth_3d=None,
    )

    assert resolve_part_distance_3d(outcome, "eyes") == pytest.approx(0.022)


def test_resolve_part_distance_falls_back_per_part_when_only_that_key_is_missing() -> None:
    """不変条件 11: 部位が個別に欠測したフレームでも、他の部位は動き続ける。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.011, eyes=0.022, nose=0.033, mouth=0.024, overall=0.018
        ),
        smooth_3d=0.018,
        part_smooth_3d={"contour": 0.014, "eyes": None, "nose": 0.019, "mouth": 0.015},
    )

    assert resolve_part_distance_3d(outcome, "eyes") == pytest.approx(0.022)
    assert resolve_part_distance_3d(outcome, "contour") == pytest.approx(0.014)


def test_resolve_part_distance_is_missing_without_a_three_dimensional_result() -> None:
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)

    assert resolve_part_distance_3d(outcome, "nose") is None


# --- §14.3e 品質ゲートの色 --------------------------------------------------


def test_every_gate_status_has_a_colour() -> None:
    """ゲートへ状態を追加したときに、KeyError へ落ちてはならない。"""
    for status in Status:
        assert gate_color(AnalysisState(status=status)) == GATE_COLORS[status]


def test_gate_colours_are_valid_hex() -> None:
    for colour in GATE_COLORS.values():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colour), colour


def test_a_missing_state_reads_as_the_neutral_colour() -> None:
    """待機中の表示が「条件を満たしている」色を借りてはならない。"""
    assert gate_color(None) == GATE_COLORS[Status.ERROR]


def test_the_blocking_states_are_visually_distinct_from_ok() -> None:
    """§5.4: 色は 1 つの伝達手段にすぎないが、それでも区別できなければならない。"""
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.HOLD]
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.NO_FACE]
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.ERROR]


# --- FR-039（撤回、2026-09-03）: 参照平均の併記 --------------------------------


def test_readings_has_no_baseline_symbols() -> None:
    """FR-039 は撤回された（§0 項目 34）。ui.readings から併記の道具が消えたことを固定する。"""
    from face_symmetry_lab.ui import readings

    for gone in (
        "BASELINE_CAVEAT",
        "baseline_note",
        "format_baseline_score",
        "format_baseline_symmetry",
    ):
        assert not hasattr(readings, gone), gone


# --- §0 項目 34（撤回、2026-09-03）: 「全体のスコア」--------------------------


def test_readings_has_no_score_symbols() -> None:
    """§8.6.7 の表示スケールは撤回された。ui.readings から道具が消えたことを固定する。"""
    from face_symmetry_lab.ui import readings

    for gone in (
        "format_score",
        "score_points_value",
        "resolve_equal_weight_distance",
        "SCORE_POINTS_TEMPLATE",
    ):
        assert not hasattr(readings, gone), gone


# --- FR-040 / §5.4: 状態と平滑化時間のテキスト表示 ----------------------------


def test_every_gate_status_has_a_text_label() -> None:
    """§5.4: 枠色を状態の唯一の手掛かりにしない。

    罠: 状態を 1 つ足したときに既定へ落ちる実装だと、その状態のときだけ HUD が
    色しか語らなくなる。`GATE_COLORS` と同じく全メンバーを要求する。
    """
    labels = {status: gate_label(AnalysisState(status=status)) for status in Status}

    assert set(labels) == set(Status)
    assert all(label for label in labels.values())
    # 「計測条件を満たしている」状態は、他のどの状態とも別の文言である。
    assert len({*labels.values()}) == len(Status)


def test_the_gate_label_without_an_analysis_reads_as_waiting() -> None:
    assert gate_label(None) == "待機中"


@pytest.mark.parametrize("banned", ["良好", "不良", "美しい", "良い", "悪い", "優れ", "理想", "劣"])
def test_the_gate_labels_carry_no_value_judgement(banned: str) -> None:
    """§5.2: 状態は「何が起きているか」であって、顔の良し悪しではない。"""
    for status in Status:
        assert banned not in gate_label(AnalysisState(status=status))


def test_the_smoothing_window_is_shown_with_its_unit() -> None:
    assert format_smoothing_window(0.4) == "平滑化 0.4 秒"
    assert format_smoothing_window(2.0) == "平滑化 2.0 秒"


def test_an_unknown_smoothing_window_is_blank_rather_than_missing() -> None:
    """`—` は「あるはずの測定値が欠測」の記号（不変条件 11）。設定の不在は別物。"""
    assert format_smoothing_window(None) == ""


# --- FR-041: 表示ラッチへ渡す鍵と、更新間隔の整形 ------------------------------


def test_the_display_signature_ignores_the_values_themselves() -> None:
    """罠: 値を鍵に含める実装は毎フレーム鍵が変わり、保持がまったく効かない。"""
    first = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.010, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )
    second = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.011, eyes=0.022, nose=0.031, mouth=0.016, overall=0.019
        ),
        smooth_3d=0.024,
    )

    assert display_signature(first) == display_signature(second)


def test_the_display_signature_changes_when_a_reading_goes_missing() -> None:
    """AC-HOLD-02: 欠測への遷移は鍵の変化として必ず現れる。"""
    measuring = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.010, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )
    partial = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=None, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )

    assert display_signature(measuring) != display_signature(partial)


def test_the_display_signature_separates_no_analysis_from_a_measured_frame() -> None:
    measured = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.010, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )

    assert display_signature(None) != display_signature(measured)


def test_the_display_signature_follows_the_gate_status() -> None:
    """品質ゲートが閉じた瞬間も、値の有無と同じく保持を打ち切る手掛かりになる。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.010, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )
    holding = dataclasses.replace(outcome, state=AnalysisState(status=Status.HOLD))

    assert display_signature(outcome) != display_signature(holding)


def test_the_display_hold_is_shown_with_its_unit() -> None:
    assert format_display_hold(2.0) == "更新 2.0 秒ごと"


@pytest.mark.parametrize("value", [0.0, None])
def test_updating_every_frame_is_not_announced(value: float | None) -> None:
    """既定の挙動を書くと HUD に常時 1 行増えるだけで、何も伝わらない。"""
    assert format_display_hold(value) == ""


# --- FR-031: gate_guidance — HUD へ出す主要因の指示文 -------------------------


def test_gate_guidance_is_empty_when_there_is_no_analysis_yet() -> None:
    """`None`（待機中）は空文字を返す。CSS の `:empty` が行ごと畳む。"""
    assert gate_guidance(None) == ""


def test_gate_guidance_is_empty_when_the_gate_is_satisfied() -> None:
    """`Status.OK` 中は測定値が出るので指示文は出ない。`—` ではなく空文字。"""
    assert gate_guidance(AnalysisState(status=Status.OK)) == ""


def test_gate_guidance_returns_pitch_instruction_for_off_axis_pitch() -> None:
    """HOLD + reason=OFF_AXIS_PITCH → REASON_GUIDANCE の対応文言（JA 既定で素通し）。"""
    state = AnalysisState(
        status=Status.HOLD,
        reason=ReasonCode.OFF_AXIS_PITCH,
        reason_codes=(ReasonCode.OFF_AXIS_PITCH,),
    )
    assert gate_guidance(state) == REASON_GUIDANCE[ReasonCode.OFF_AXIS_PITCH]


def test_gate_guidance_returns_no_face_instruction_for_no_face_status() -> None:
    """`Status.NO_FACE + reason=NO_FACE` → `REASON_GUIDANCE[NO_FACE]`。"""
    state = AnalysisState(
        status=Status.NO_FACE,
        reason=ReasonCode.NO_FACE,
        reason_codes=(ReasonCode.NO_FACE,),
    )
    assert gate_guidance(state) == REASON_GUIDANCE[ReasonCode.NO_FACE]


def test_gate_guidance_falls_back_to_reason_codes_when_reason_is_none() -> None:
    """HOLD + reason=None, reason_codes=(TOO_FAR,) → `reason_codes[0]` を使う。

    品質ゲートが `reason` を省略しても `reason_codes` に記録を残す運用への対応。
    """
    state = AnalysisState(
        status=Status.HOLD,
        reason=None,
        reason_codes=(ReasonCode.TOO_FAR,),
    )
    assert gate_guidance(state) == REASON_GUIDANCE[ReasonCode.TOO_FAR]


def test_gate_guidance_returns_the_primary_reason_when_multiple_codes_exist() -> None:
    """HOLD + reason=OFF_AXIS_YAW, reason_codes=(OFF_AXIS_YAW, OFF_AXIS_PITCH)
    → yaw（`reason` フィールド）が主要因として選ばれる。
    """
    state = AnalysisState(
        status=Status.HOLD,
        reason=ReasonCode.OFF_AXIS_YAW,
        reason_codes=(ReasonCode.OFF_AXIS_YAW, ReasonCode.OFF_AXIS_PITCH),
    )
    assert gate_guidance(state) == REASON_GUIDANCE[ReasonCode.OFF_AXIS_YAW]


def test_gate_guidance_pitch_text_differs_from_yaw_text() -> None:
    """罠: YAW と PITCH の文言を取り違えると写経実装でもこれが通ってしまう。"""
    assert REASON_GUIDANCE[ReasonCode.OFF_AXIS_PITCH] != REASON_GUIDANCE[ReasonCode.OFF_AXIS_YAW]
    state = AnalysisState(
        status=Status.HOLD,
        reason=ReasonCode.OFF_AXIS_PITCH,
        reason_codes=(ReasonCode.OFF_AXIS_PITCH,),
    )
    assert gate_guidance(state) != REASON_GUIDANCE[ReasonCode.OFF_AXIS_YAW]


def test_gate_guidance_always_returns_empty_or_a_known_guidance_string() -> None:
    """返り値は `""` か `REASON_GUIDANCE` の値のいずれかだけ。第三の値を作らない。"""
    known = set(REASON_GUIDANCE.values())
    for code in ReasonCode:
        state = AnalysisState(
            status=Status.HOLD,
            reason=code,
            reason_codes=(code,),
        )
        result = gate_guidance(state)
        assert result in known or result == "", f"{code}: 未知の戻り値 {result!r}"


# --- FR-031/§6.2: gate_warning — 非ブロッキング助言の HUD 表示 ---------------


def test_gate_warning_is_empty_when_there_is_no_analysis_yet() -> None:
    """`None`（待機中）は空文字を返す。CSS の `:empty` が行ごと畳む。"""
    assert gate_warning(None) == ""


def test_gate_warning_is_empty_when_warning_codes_is_empty() -> None:
    """`warning_codes` が空のとき、status に関わらず空文字を返す。"""
    assert gate_warning(AnalysisState(status=Status.OK)) == ""


def test_gate_warning_returns_low_light_instruction_when_ok_with_warning() -> None:
    """`Status.OK` 中でも `warning_codes=(LOW_LIGHT,)` があれば助言を返す。

    `gate_guidance` と違い、status 判定を行わない——OK 中でも暗ければ助言を出す。
    """
    state = AnalysisState(status=Status.OK, warning_codes=(ReasonCode.LOW_LIGHT,))
    assert gate_warning(state) == REASON_GUIDANCE[ReasonCode.LOW_LIGHT]


def test_gate_warning_returns_overexposed_instruction_for_overexposed() -> None:
    """`warning_codes=(OVEREXPOSED,)` → `REASON_GUIDANCE[OVEREXPOSED]`（既定 JA で素通し）。"""
    state = AnalysisState(status=Status.OK, warning_codes=(ReasonCode.OVEREXPOSED,))
    assert gate_warning(state) == REASON_GUIDANCE[ReasonCode.OVEREXPOSED]


def test_gate_warning_does_not_show_blocking_reason_as_warning() -> None:
    """罠: HOLD かつ `reason=OFF_AXIS_PITCH`・`warning_codes=()` のとき
    `gate_warning` は空文字——阻害理由を警告として出さない。"""
    state = AnalysisState(
        status=Status.HOLD,
        reason=ReasonCode.OFF_AXIS_PITCH,
        reason_codes=(ReasonCode.OFF_AXIS_PITCH,),
        warning_codes=(),
    )
    assert gate_warning(state) == ""


def test_gate_warning_and_gate_guidance_are_independent_for_ok_with_low_light() -> None:
    """罠: OK かつ `warning_codes=(LOW_LIGHT,)` のとき——`gate_guidance` は空文字
    だが `gate_warning` は非空。2 つの関数が独立していることを固定する。"""
    state = AnalysisState(status=Status.OK, warning_codes=(ReasonCode.LOW_LIGHT,))
    assert gate_guidance(state) == ""
    assert gate_warning(state) != ""
