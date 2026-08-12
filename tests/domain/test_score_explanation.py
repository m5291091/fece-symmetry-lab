"""§8.6.3（H4）「全体のスコア」の計算過程を組み立てる純関数。

ここが守るのは 1 点に尽きる。**計算過程は既存の式を経由するのであって、
表示のために式を書き直したものではない。** `ui` が読む DTO の値が
`equal_weight_distance_from_parts`（§8.6.5）と `score_points`（§8.6.7）から
一歩でもずれたら、画面に出る計算過程が実際の指標と別物になる。

罠の置き方: 途中経過（二乗和・平均・平方根）を独立に計算して並べる実装は
`test_the_intermediate_steps_close_back_onto_the_combined_distance` で落ちる。
欠測を 0 で埋める実装は `test_a_single_missing_part_makes_the_whole_explanation_missing`
で落ちる（不変条件 11）。
"""

from __future__ import annotations

import math

import pytest

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.score_explanation import explain_score
from face_symmetry_lab.domain.score_scale import (
    DISTANCE_AT_FULL_SCORE,
    DISTANCE_AT_ZERO_SCORE,
    score_points,
)
from face_symmetry_lab.domain.symmetry_3d import equal_weight_distance_from_parts

#: 4 部位が揃った代表入力。どの値も §8.6.7 のアンカー表記（1.00% / 5.00%）と
#: 衝突しない桁にしてある。衝突すると、画面へ出た数字がアンカーなのか実測値
#: なのかを UI 側のテストが区別できない。
SAMPLE_DISTANCES: dict[str, float] = {
    "contour": 0.0211,
    "eyes": 0.0144,
    "nose": 0.0062,
    "mouth": 0.0183,
}


def test_the_parts_are_listed_in_the_versioned_order() -> None:
    """部位の並びは `PART_NAMES` が正本である。UI 側で並べ直さない。"""
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert tuple(part.name for part in explanation.parts) == PART_NAMES


def test_each_part_carries_its_own_distance_and_square() -> None:
    """ステップ 1 が示すのは、部位ごとの「ずれ」と、次で使う二乗である。"""
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    for part in explanation.parts:
        assert part.distance == pytest.approx(SAMPLE_DISTANCES[part.name])
        assert part.square == pytest.approx(SAMPLE_DISTANCES[part.name] ** 2)


def test_the_combined_distance_is_the_versioned_equal_weight_distance() -> None:
    """§8.6.5: `s_3d` は既存の純関数から来る。

    罠: 表示用に `sqrt(sum/4)` を書き直した実装は、部位数の定義が
    `PART_NAMES` から外れた瞬間に本体と食い違う。
    """
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert explanation.combined_distance == pytest.approx(
        equal_weight_distance_from_parts(SAMPLE_DISTANCES)
    )


def test_the_points_come_from_the_versioned_display_scale() -> None:
    """§8.6.7: 点は `score_points` を経由する。UI が 2 つ目の目盛りを持たない。"""
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert explanation.points == pytest.approx(score_points(explanation.combined_distance))
    # sqrt(mean(2.11² + 1.44² + 0.62² + 1.83²)) = 1.60% → 100×(5.00-1.60)/(5.00-1.00)
    assert explanation.points == pytest.approx(85.0, abs=0.05)


def test_the_intermediate_steps_close_back_onto_the_combined_distance() -> None:
    """ステップ 2 の途中経過は、最終値と同じ 1 本の計算でなければならない。

    罠: 二乗和・平均・平方根を別々に計算して見栄えのために並べた実装は、
    丸めや部位数の取り違えでこの恒等式が閉じなくなる。画面に出る式が
    「その通りに計算しても答えが合わない式」になるという壊れ方をする。
    """
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert explanation.square_sum == pytest.approx(sum(part.square for part in explanation.parts))
    assert explanation.mean_square == pytest.approx(explanation.square_sum / len(explanation.parts))
    assert math.sqrt(explanation.mean_square) == pytest.approx(explanation.combined_distance)


def test_the_anchors_are_read_from_the_frozen_display_scale() -> None:
    """§8.6.7: アンカーは版に凍結された定数であり、DTO はその写しを作らない。"""
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert explanation.distance_at_full_score == DISTANCE_AT_FULL_SCORE
    assert explanation.distance_at_zero_score == DISTANCE_AT_ZERO_SCORE


def test_the_slope_is_derived_from_the_anchors_rather_than_hardcoded() -> None:
    """§8.6.7: ずれが 1 ポイント動くと点は 25 点動く。

    罠: `25.0` を直接書いた実装は、アンカーを差し替えた版
    （`display-scale/2.0.0`）で黙って嘘になる。ここでは `score_points` の
    実際の傾きと一致することを要求する。
    """
    explanation = explain_score(SAMPLE_DISTANCES)

    assert explanation is not None
    assert explanation.points_per_deviation_percent == pytest.approx(25.0)

    before = score_points(0.020)
    after = score_points(0.030)
    assert before is not None and after is not None
    assert before - after == pytest.approx(explanation.points_per_deviation_percent * 1.0)


@pytest.mark.parametrize("missing", PART_NAMES)
def test_a_single_missing_part_makes_the_whole_explanation_missing(missing: str) -> None:
    """§8.6.5・不変条件 11: 使える部位だけで平均しない。0 でも埋めない。

    罠: 欠測を 0 へ落とす実装は、ここで `s_3d` が小さくなり「100.0 点」という
    偽の上限を画面へ出す。
    """
    with_none = dict(SAMPLE_DISTANCES) | {missing: None}
    without_key = {name: value for name, value in SAMPLE_DISTANCES.items() if name != missing}

    assert explain_score(with_none) is None
    assert explain_score(without_key) is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_part_is_reported_as_missing(value: float) -> None:
    """不変条件 11: NaN / Inf を計算過程へ通さない。

    `equal_weight_distance_from_parts` は有限性を検査しないため、NaN はそのまま
    二乗和を汚染し、画面に `nan%` という行が出る。ここで弾く。
    """
    assert explain_score(dict(SAMPLE_DISTANCES) | {"nose": value}) is None


def test_an_empty_mapping_is_missing_rather_than_a_perfect_score() -> None:
    """部位が 1 つも無い入力は欠測である。空の平均は 0（= 100 点）ではない。"""
    assert explain_score({}) is None
