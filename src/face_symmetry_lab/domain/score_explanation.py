"""「全体のスコア」の計算過程（§8.6.3 の H4 ステップ開示）。

`ui` が画面へ並べる 3 ステップ——4 部位のずれを読む / 1 つにまとめる / 点へ
置き換える——の**中身を、表示から独立した値として組み立てる**。

**新しい指標を導入しない。** `combined_distance` は §8.6.5 の
`equal_weight_distance_from_parts`、`points` は §8.6.7 の `score_points` を
そのまま経由する。ここに置くのは、その 1 本の計算を利用者が追えるように
途中経過（部位ごとの二乗・二乗和・平均）へ分解した形だけである。

**`domain` に置く理由。** 途中経過を `ui` で組み立てると、画面表示のためだけの
2 つ目の計算経路ができる（CLAUDE.md §5「`ui` に推論式・幾何計算を実装しない」）。
そちらが本体からずれても、画面には「その通り計算しても答えが合わない式」が
出るだけで、テストが無ければ誰も気付かない。

**欠測は欠測のまま返す。** 部位が 1 つでも欠けていれば、あるいは非有限値が
混ざっていれば、DTO 全体が `None` になる。使える部位だけで平均しない
（§8.6.5）、0 で埋めない（不変条件 11）。0 点は「ずれ 5.0% 以上」という測定
結果であり、値が無いこととは別である。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.score_scale import (
    DISTANCE_AT_FULL_SCORE,
    DISTANCE_AT_ZERO_SCORE,
    POINTS_PER_DEVIATION_PERCENT,
    score_points,
)
from face_symmetry_lab.domain.symmetry_3d import equal_weight_distance_from_parts


@dataclass(frozen=True, slots=True)
class PartDeviation:
    """ステップ 1 の 1 行。部位 1 つのずれと、ステップ 2 で使う二乗。"""

    #: `PART_NAMES` の識別子。表示ラベル（「輪郭」等）は `ui` が持つ。
    name: str
    #: `d_3d,p`。両眼 anchor 間距離を 1.0 とした無次元の距離。
    distance: float
    #: `d_3d,p²`。ステップ 2 の二乗和へ入る項そのもの。
    square: float


@dataclass(frozen=True, slots=True)
class ScoreExplanation:
    """3 ステップを通した計算過程の全体。`ui` はこれを整形して並べるだけ。"""

    #: ステップ 1。`PART_NAMES` の順に 4 件。
    parts: tuple[PartDeviation, ...]
    #: ステップ 2 の途中経過: 二乗和。
    square_sum: float
    #: ステップ 2 の途中経過: 部位数で割った平均。
    mean_square: float
    #: ステップ 2 の結果 = §8.6.5 の `s_3d`。`sqrt(mean_square)` に一致する。
    combined_distance: float
    #: ステップ 3 の結果 = §8.6.7 の `score_points`。単位は「点」であって % ではない。
    points: float
    #: §8.6.7 のアンカー（100 点側）。`domain.score_scale` の凍結値をそのまま運ぶ。
    distance_at_full_score: float
    #: §8.6.7 のアンカー（0 点側）。同上。
    distance_at_zero_score: float
    #: ずれが 1 ポイント（両眼間隔の 1%）動いたときに点が動く量。アンカー 2 点
    #: から導き、目盛りの粗さを利用者へ伝える注記の根拠になる。
    points_per_deviation_percent: float


def explain_score(distances: Mapping[str, float | None]) -> ScoreExplanation | None:
    """部位別距離から、§8.6.3（H4）の 3 ステップ分の値を組み立てる。

    ``distances`` は呼び出し側が解決済みの部位別 `d_3d,p`（`ui` では
    `resolve_part_distance_3d` が返す主表示の値）である。**ここでは平滑値と
    生値を区別しない。** どちらを渡すかは表示側の決定であり、計算は同じである。

    4 部位が有限値で揃っているときだけ DTO を返す。1 つでも欠測・キー欠落・
    非有限なら `None`。
    """
    parts: list[PartDeviation] = []
    for name in PART_NAMES:
        value = distances.get(name)
        if value is None or not math.isfinite(value):
            return None
        parts.append(PartDeviation(name=name, distance=value, square=value**2))

    # §8.6.5 の純関数を通す。ここで `sqrt(sum/4)` を書き直すと、部位集合の定義が
    # 2 か所へ分かれる。
    combined = equal_weight_distance_from_parts(distances)
    points = score_points(combined)
    if combined is None or points is None:
        return None

    square_sum = sum(part.square for part in parts)
    return ScoreExplanation(
        parts=tuple(parts),
        square_sum=square_sum,
        mean_square=square_sum / len(parts),
        combined_distance=combined,
        points=points,
        distance_at_full_score=DISTANCE_AT_FULL_SCORE,
        distance_at_zero_score=DISTANCE_AT_ZERO_SCORE,
        points_per_deviation_percent=POINTS_PER_DEVIATION_PERCENT,
    )
