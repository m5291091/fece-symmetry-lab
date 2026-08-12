"""§8.6.7 の表示スケール（`display-scale/1.0.0`、AC-DISPLAY-05）。

罠の置き方: §8.6.3 の `100 × (1 - d)` をそのまま返す実装は
`test_the_measured_median_is_stretched_away_from_the_ninety_eight_percent_band`
で落ちる。実測の中央値 `d = 0.01720` が 98.28 のままになるためである。
アンカーを取り違えた実装（`D_FULL` と `D_ZERO` の入れ替え、100 倍忘れ）は
アンカー自身の 2 件で落ちる。
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from face_symmetry_lab.domain.score_scale import (
    DISPLAY_SCALE_VERSION,
    DISTANCE_AT_FULL_SCORE,
    DISTANCE_AT_ZERO_SCORE,
    POINTS_PER_DEVIATION_PERCENT,
    score_points,
)

#: `docs/symmetry-metric-measurements.md` §14.4 の 3 セッション（被写体 1 名）の
#: 表示中央値 98.280 / 98.249 / 98.523pt を距離へ戻したもの。
MEASURED_MEDIANS: tuple[float, ...] = (0.01720, 0.01751, 0.01477)


def test_the_versioned_anchors_are_frozen_at_the_documented_values() -> None:
    """§8.6.7: アンカーは版に凍結された固定値であり、実行時に導出しない。"""
    assert DISPLAY_SCALE_VERSION == "display-scale/1.0.0"
    assert pytest.approx((0.010, 0.050)) == (DISTANCE_AT_FULL_SCORE, DISTANCE_AT_ZERO_SCORE)


def test_the_full_score_anchor_maps_to_one_hundred_points() -> None:
    assert score_points(DISTANCE_AT_FULL_SCORE) == pytest.approx(100.0)


def test_the_zero_score_anchor_maps_to_zero_points() -> None:
    assert score_points(DISTANCE_AT_ZERO_SCORE) == pytest.approx(0.0)


def test_the_measured_median_is_stretched_away_from_the_ninety_eight_percent_band() -> None:
    """§8.6.7 の主目的。実測の中央値が 98 付近ではなく、目盛りの中ほどへ出る。

    §8.6.3 の変換をそのまま返す実装（`100 × (1 - d)` = 98.28）はここで落ちる。
    """
    assert score_points(MEASURED_MEDIANS[0]) == pytest.approx(82.0)
    assert score_points(MEASURED_MEDIANS[1]) == pytest.approx(81.225)
    assert score_points(MEASURED_MEDIANS[2]) == pytest.approx(88.075)


def test_the_three_sessions_of_one_subject_span_the_documented_point_width() -> None:
    """§8.6.7: セッション間全幅 0.273pt は 6.8 点に対応する。

    この幅は「同一人物を撮り直しただけで動く量」であり、UI が注記する根拠でも
    ある。目盛りを 25 倍にすると、ばらつきも同じ 25 倍で拡大する。
    """
    points = [score_points(distance) for distance in MEASURED_MEDIANS]
    assert None not in points
    width = max(points) - min(points)  # type: ignore[type-var]
    assert width == pytest.approx(6.85, abs=0.05)


def test_a_larger_deviation_never_earns_more_points() -> None:
    """単調非増加。ずれが増えて点が上がる向きは、指標の符号が壊れている。"""
    distances = [0.005, 0.010, 0.0172, 0.030, 0.050, 0.080]
    points = [score_points(distance) for distance in distances]

    assert None not in points
    assert all(earlier >= later for earlier, later in pairwise(points))  # type: ignore[operator]


def test_points_stay_inside_the_scale_at_both_ends() -> None:
    """§8.6.7: 両端は clamp する。`s_3d` 自体は上限なく記録され続ける。"""
    assert score_points(0.0) == pytest.approx(100.0)
    assert score_points(0.002) == pytest.approx(100.0)
    assert score_points(0.20) == pytest.approx(0.0)
    assert score_points(1.5) == pytest.approx(0.0)


def test_the_scale_is_linear_and_reversible_from_the_two_anchors() -> None:
    """§8.6.7・AC-DISPLAY-05: アンカー 2 点から `s_3d` へ一意に戻せる。

    逆変換は production へ置かない（利用者が読むのは順方向だけである）。ここでは
    公開している 2 つのアンカーだけで逆算し、clamp の外側で往復が閉じることを
    確かめる。指数変換のような非線形写像を入れるとこの往復が閉じない。
    """
    span = DISTANCE_AT_ZERO_SCORE - DISTANCE_AT_FULL_SCORE
    for distance in (0.012, 0.0172, 0.025, 0.040):
        points = score_points(distance)
        assert points is not None
        recovered = DISTANCE_AT_ZERO_SCORE - span * points / 100.0
        assert recovered == pytest.approx(distance)


def test_the_published_slope_matches_the_actual_scale() -> None:
    """§8.6.3 のステップ 3 が画面へ書く「1 ポイントで 25 点」の根拠。

    罠: UI 側や注記へ `25` を直書きした実装は、アンカーを差し替えた版
    （`display-scale/2.0.0`）で画面の説明だけが黙って嘘になる。定数が実際の
    傾きと一致していることをここで固定する。
    """
    assert pytest.approx(25.0) == POINTS_PER_DEVIATION_PERCENT

    before = score_points(0.020)
    after = score_points(0.030)
    assert before is not None and after is not None
    assert before - after == pytest.approx(POINTS_PER_DEVIATION_PERCENT)


def test_a_missing_distance_stays_missing_instead_of_scoring_zero() -> None:
    """不変条件 11: 欠測を 0 点で埋めない。0 点は「ずれ 5.0% 以上」の測定結果。"""
    assert score_points(None) is None
    assert score_points(DISTANCE_AT_ZERO_SCORE) == pytest.approx(0.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_distance_is_reported_as_missing(value: float) -> None:
    """不変条件 11: NaN / Inf を点へ通さない。

    `min`/`max` による clamp は NaN を黙って 0.0 へ落とすため、clamp より前に
    弾く必要がある。0.0 になると「ずれ 5.0% 以上」という測定結果と区別が
    付かなくなる。
    """
    assert score_points(value) is None
