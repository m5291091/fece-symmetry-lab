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

from face_symmetry_lab.domain.models import AnalysisState, GenderLabel, Status
from face_symmetry_lab.domain.reference_baseline import BaselineReading
from face_symmetry_lab.ui.readings import (
    BASELINE_CAVEAT,
    GATE_COLORS,
    baseline_note,
    display_signature,
    format_baseline_score,
    format_baseline_symmetry,
    format_display_hold,
    format_distance,
    format_score,
    format_smoothing_window,
    format_symmetry,
    gate_color,
    gate_label,
    resolve_equal_weight_distance,
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


# --- §8.6.7 「全体のスコア」の点 ----------------------------------------------


def test_the_score_formatter_reports_points_and_never_a_percent_sign() -> None:
    """§8.6.7: 単位は「点」であり %ではない。対称度 % と同じ数字として読ませない。

    罠: `format_symmetry` を流用した実装（`98.28%`）はここで落ちる。
    """
    assert format_score(0.0172) == "82.0 点"
    assert "%" not in format_score(0.0172)
    assert format_score(None) == "—"


def test_a_missing_score_is_not_shown_as_zero_points() -> None:
    """不変条件 11・§8.6.7: 0 点は「ずれ 5.0% 以上」という測定結果であり、欠測とは
    別である。両者を同じ表示にしてはならない。"""
    assert format_score(None) == "—"
    assert format_score(0.050) == "0.0 点"
    assert format_score(0.080) == "0.0 点"


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


def test_the_resolved_equal_weight_distance_always_reads_the_raw_value() -> None:
    """§8.6.5/§8.7: `s_3d` は EMA の対象外なので、平滑値へ落ちる経路を持たない。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.0172,
    )

    symmetry_3d = outcome.symmetry_3d
    assert symmetry_3d is not None
    assert resolve_equal_weight_distance(outcome) == pytest.approx(
        symmetry_3d.equal_weight_distance
    )
    # `d_3d`（平滑値・生値のいずれ）とも別の値であることを固定する。
    assert resolve_equal_weight_distance(outcome) != pytest.approx(0.0172)


def test_the_resolved_equal_weight_distance_is_missing_without_a_three_dimensional_result() -> None:
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)

    assert resolve_equal_weight_distance(outcome) is None


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


# --- FR-039 参照平均の併記 ----------------------------------------------------


def test_baseline_symmetry_shows_the_mean_and_the_signed_difference() -> None:
    reading = BaselineReading(mean=96.31, delta=2.11)
    assert format_baseline_symmetry(reading) == "平均 96.31%（+2.11）"


def test_baseline_symmetry_keeps_the_sign_when_the_value_is_below_the_mean() -> None:
    """罠: `abs()` を挟んだ実装は符号を落とし、どちら側かが読めなくなる。"""
    assert (
        format_baseline_symmetry(BaselineReading(mean=96.31, delta=-1.4)) == "平均 96.31%（-1.40）"
    )


def test_baseline_score_uses_points_and_not_percent() -> None:
    """§8.6.7: 点と % が同じ目盛りに見えてはならない（`format_score` と同じ桁）。"""
    text = format_baseline_score(BaselineReading(mean=41.5, delta=20.8))
    assert text == "平均 41.5 点（+20.8）"
    assert "%" not in text


def test_a_missing_display_value_leaves_the_mean_without_a_difference() -> None:
    """不変条件 11: 差を 0 で埋めない（`+0.00` は「平均と同じ」という測定結果）。"""
    assert format_baseline_symmetry(BaselineReading(mean=96.31, delta=None)) == "平均 96.31%"
    assert format_baseline_score(BaselineReading(mean=41.5, delta=None)) == "平均 41.5 点"


def test_a_vanishing_difference_never_renders_as_negative_zero() -> None:
    """丸め誤差から「わずかに下」と読める表示を作らない。"""
    assert format_baseline_symmetry(BaselineReading(mean=96.31, delta=-0.001)) == (
        "平均 96.31%（+0.00）"
    )


def test_no_baseline_renders_as_nothing_rather_than_a_missing_marker() -> None:
    """`—` は測定の欠測に使う記号であり、併記の不在ではない。"""
    assert format_baseline_symmetry(None) == ""
    assert format_baseline_score(None) == ""


def test_the_baseline_note_names_the_population_and_the_count() -> None:
    """AC-BASELINE-04: 何と比べているのかを画面から読み取れるようにする。"""
    note = baseline_note(age=20, gender=GenderLabel.MALE, count=1075)
    assert "20 歳" in note
    assert "男性" in note
    assert "1,075 件" in note
    assert "人種で絞り込まず" in note


@pytest.mark.parametrize(
    "banned",
    ["偏差値", "順位", "パーセンタイル", "分位", "合格", "平均より上", "平均より下"],
)
def test_the_baseline_note_never_ranks_the_user(banned: str) -> None:
    """§10.12-1: 差は引き算であって、順序への写像ではない。"""
    note = baseline_note(age=20, gender=GenderLabel.MALE, count=75)
    assert banned not in note
    assert banned not in BASELINE_CAVEAT


@pytest.mark.parametrize("banned", ["美しい", "良い", "悪い", "優れ", "理想", "劣"])
def test_the_baseline_note_carries_no_value_judgement(banned: str) -> None:
    assert banned not in baseline_note(age=20, gender=GenderLabel.MALE, count=75)


def test_the_baseline_note_is_a_single_line() -> None:
    """§10.12-5 / §5.3.1（2026-08-16）: 但し書きは 1 行まで。

    以前はこれが吹き出しの中で測定値より長く、そのために読まれなくなっていた。
    残すのは「括弧の中が何か」と「探索値であること」、そして母集団だけである。
    根拠と制限の全文は About が持つ。
    """
    note = baseline_note(age=20, gender=GenderLabel.FEMALE, count=75)

    assert "括弧内は表示値と平均の差です" in note
    assert "探索値" in note
    assert "20 歳" in note and "75" in note
    assert "人種で絞り込まず" in note
    assert len(BASELINE_CAVEAT) < 60


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
