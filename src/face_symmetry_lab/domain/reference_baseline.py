"""FR-039 / §10.12: 参照平均の併記に必要な純粋計算。

吹き出し・HUD に出ている 6 つの数値（全体の対称度・全体のスコア・部位別 4 つ）へ、
§9.5 のセルテーブルから作った**同じ参考年齢・同じ参考性別の平均**と、表示値との差を
添えるための最小の道具だけを置く。集計そのものは `domain.reference_stats` が行い、
ここは「どのセルを読むか」と「差をどう取るか」だけを決める。

**ここに置いていないもの**は 2 つある。**(1) 順位化。** 偏差値・順位・パーセンタイル・
分位・「平均より上」「平均より下」は §10.12-1 が禁じており、この module はそれらを
計算する関数を持たない。持たせないこと自体が禁止の実装である。**(2) 人種による
絞り込み。** アプリは人種を推定せず（不変条件 7）、`baseline_filter` は `races` を
空のまま返す。

この module は streamlit / mediapipe / onnxruntime / cv2 / numpy のいずれも import
しない。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import GenderLabel
from face_symmetry_lab.domain.reference_stats import (
    MAX_AGE,
    MIN_AGE,
    MetricSummary,
    ReferenceFilter,
)

#: §8.6.7 の「全体のスコア」に対応するセル列（§9.5）。
SCORE_METRIC: Final[str] = "score_points"

#: 「全体の対称度」に対応するセル列。`d_3d` を §8.6.3 の変換で読んだ百分率である。
OVERALL_METRIC: Final[str] = "symmetry_percent_overall"

#: 部位名（`domain.metrics.PART_NAMES`）から §9.5 の列名への対応。列名が 1 文字でも
#: 違うと平均は例外ではなく欠測として消えるため、`tests/domain/test_reference_baseline.py`
#: が `METRIC_NAMES` との一致を固定している。
PART_METRICS: Final[Mapping[str, str]] = MappingProxyType(
    {name: f"symmetry_percent_{name}" for name in PART_NAMES}
)


@dataclass(frozen=True, slots=True)
class BaselineReading:
    """1 つの数値に対する併記の中身。

    `delta` は **表示値 − 平均** であり、表示値が欠測・非有限のときは `None` になる
    （0 で埋めない、不変条件 11）。`mean` は欠測を持たない——平均が無いときは
    `BaselineReading` 自体を作らない。
    """

    mean: float
    delta: float | None


def baseline_age(age_years: float) -> int | None:
    """参考年齢から、母集団として読むセルの年齢（1 歳刻み）を決める。

    **丸めは画面と同じでなければならない。** HUD と吹き出しは参考年齢を
    `f"{age:.0f} 歳"`（偶数丸め）で出すため、ここで `int()` の切り捨てや
    `floor(x + 0.5)` を使うと、ちょうど .5 のときだけ「参考年齢 27 歳」と
    「28 歳の平均」が並ぶ（§10.12-2）。`round` は同じ偶数丸めである。

    セルテーブルの範囲（§9.5、18〜75）の外と非有限値は `None`。境界へ clamp しない
    ——範囲外の顔に「18 歳の平均」を当てることになるためである。
    """
    if not math.isfinite(age_years):
        return None
    age = round(age_years)
    if not MIN_AGE <= age <= MAX_AGE:
        return None
    return age


def baseline_filter(*, age: int, gender: GenderLabel) -> ReferenceFilter:
    """§10.12-2: 母集団は「その年齢・その性別」だけで決める。

    `races` は空のまま返す。空の軸は絞り込みに使われない（`select_cells`）ので、
    結果は全人種の合算になる。**人種で絞る経路を作らない**のは、アプリが人種を
    推定しないためである（不変条件 7）。絞れないのではなく、絞る手段を持たない
    ことが正しい状態である。
    """
    return ReferenceFilter(genders=frozenset({gender.value}), age_range=(age, age))


def baseline_reading(
    metrics: Mapping[str, MetricSummary],
    metric_name: str,
    *,
    value: float | None,
) -> BaselineReading | None:
    """1 指標分の平均と、表示値との差を組み立てる。

    合計件数が §9.5 のしきい値未満のとき `ReferenceAggregate.metrics` は空になるため
    （`combine`）、この関数も自然に `None`——併記なし——へ落ちる。しきい値の判定を
    ここで書き直さない。
    """
    summary = metrics.get(metric_name)
    if summary is None:
        return None
    if value is None or not math.isfinite(value):
        return BaselineReading(mean=summary.mean, delta=None)
    return BaselineReading(mean=summary.mean, delta=value - summary.mean)
