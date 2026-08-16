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
import statistics
from collections.abc import Iterable, Mapping, Sequence
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


@dataclass(frozen=True, slots=True)
class ObservedAges:
    """FR-039（2026-08-16）: 1 セッションで観測した参考年齢の範囲。

    `low`〜`high` は**四分位範囲（真ん中 50%）**を 1 歳へ丸めて外側へ広げたもので、
    観測の min〜max ではない。1 フレームの外れ値（例: 1 回だけ 55 歳）で母集団が
    不当に広がるのを避けるためである。要約（FR-021）が「変動幅（真ん中 50%）」を
    使うのと同じ語・同じ理由にそろえてある。

    `median` は母集団として画面へ出す代表値である。範囲だけを出すと「20〜28 歳の
    平均」がどのあたりの顔に対するものか読み取れない。
    """

    median: int
    low: int
    high: int
    sample_count: int


def observed_ages(age_years: Sequence[float]) -> ObservedAges | None:
    """観測した参考年齢の列から、母集団に使う範囲を決める。

    各値は `baseline_age` と**同じ丸め**を通す（画面の「27 歳」と母集団の年齢が
    食い違わないようにするため、§10.12-2）。§9.5 の範囲外（18〜75 の外）と非有限値は
    そこで落ちる——clamp しない。落とした後に 1 件も残らなければ `None` を返す。

    四分位は `statistics.quantiles` の補間を使い、`low` は切り下げ、`high` は
    切り上げで整数へ寄せる。**外側へ広げる**のは、丸めで母集団が実際の観測より
    狭くなることを避けるためである。標本が 1 件のときは、その 1 歳がそのまま範囲になる。
    """
    ages = [age for age in (baseline_age(value) for value in age_years) if age is not None]
    if not ages:
        return None
    if len(ages) == 1:
        only = ages[0]
        return ObservedAges(median=only, low=only, high=only, sample_count=1)
    first, _second, third = statistics.quantiles(ages, n=4, method="inclusive")
    low = max(MIN_AGE, math.floor(first))
    high = min(MAX_AGE, math.ceil(third))
    return ObservedAges(
        median=round(statistics.median(ages)),
        low=min(low, high),
        high=max(low, high),
        sample_count=len(ages),
    )


def session_baseline_filter(
    *,
    ages: ObservedAges,
    genders: Iterable[GenderLabel],
) -> ReferenceFilter:
    """§9.6.1: セッション全体を 1 つの母集団として読むための絞り込み。

    `baseline_filter` の一般化である。あちらが `age_range=(age, age)` と 1 つの
    性別で 1 点を指すのに対し、こちらは範囲と性別の集合を指す。**性別は観測された
    ラベルの和集合**にする——判定が途中で入れ替わったなら、その両方がそのセッション
    で表示された判定だからである。

    `races` を空のまま返すのは `baseline_filter` と同じ理由で、アプリが人種を推定
    しないためである（不変条件 7）。
    """
    return ReferenceFilter(
        genders=frozenset(gender.value for gender in genders),
        age_range=(ages.low, ages.high),
    )
