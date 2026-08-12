"""`ui.readings`: 表示値をどこから読み、どう整形するか。

`ui.scores`（描画）から分離した層に対応する。ここで守るのは 3 点である。

1. **同じ測定値に 2 つ目の表示規則を作らない。** HUD・吹き出し・詳細パネルは
   すべてこの module を通る。
2. **主指標は 3D である**（§8.6.3、項目 22）。fixture は 2D 側へ大きく離れた値
   （0.5）を入れてあるので、誤って 2D を読む実装はここで落ちる。
3. **欠測は `—`。** 0 でも前回値でもない（不変条件 11、NFR-013）。
"""

from __future__ import annotations

import re

import pytest

from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.ui.readings import (
    GATE_COLORS,
    format_distance,
    format_score,
    format_symmetry,
    gate_color,
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
