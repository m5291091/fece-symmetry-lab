"""§8.6.3 の対称度表示変換（`domain.symmetry_display.symmetry_percent`）。

**撤回**（2026-09-03、§0 項目 34）: §8.6.7 の表示スケール（`display-scale/1.0.0`、
旧 `domain.score_scale.score_points`）を機能ごと削除した。旧
`tests/domain/test_score_scale.py` は本ファイルへ改名し、残るのは
`symmetry_percent` のテストだけである。

罠の置き方: `100 × (1 - d)` を clamp より先に NaN/Inf へ適用する実装は
`test_a_non_finite_distance_is_reported_as_missing` で落ちる（`min`/`max` は
NaN を黙って 0.0 へ落とすため）。
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from face_symmetry_lab.domain.symmetry_display import symmetry_percent


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 100.0),
        (0.0215, 97.85),
        (0.5, 50.0),
        (1.0, 0.0),
    ],
)
def test_symmetry_percent_is_the_complement_of_the_distance(
    distance: float, expected: float
) -> None:
    """§8.6.3: `100% - ずれ%`。係数のない引き算だけの表示変換。"""
    assert symmetry_percent(distance) == pytest.approx(expected)


def test_a_deviation_beyond_the_interocular_distance_floors_at_zero() -> None:
    """`d` 自体は上限を持たないが、表示は 0% で下げ止まる（検出破綻に相当）。"""
    assert symmetry_percent(4.0) == pytest.approx(0.0)


def test_the_transform_is_monotonically_non_increasing() -> None:
    """ずれが大きいほど対称度は下がるか、変わらない。符号が壊れていないこと。"""
    distances = [0.0, 0.02, 0.1, 0.31, 0.5, 1.0]
    percents = [symmetry_percent(distance) for distance in distances]

    assert None not in percents
    assert all(earlier >= later for earlier, later in pairwise(percents))  # type: ignore[operator]


def test_a_missing_distance_is_reported_as_missing() -> None:
    """不変条件 11: 欠測を対称度 100%（完全対称）として読ませない。"""
    assert symmetry_percent(None) is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_distance_is_reported_as_missing(value: float) -> None:
    """不変条件 11: NaN / Inf を対称度へ通さない。

    `min`/`max` による clamp は NaN を黙って 0.0 へ落とすため、clamp より前に
    弾く必要がある。
    """
    assert symmetry_percent(value) is None
