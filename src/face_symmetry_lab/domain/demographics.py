"""18〜75 歳の年齢・性別モデルに対する、検証付きの domain 変換。

**§9.6.1（レポートの参考年齢・参考性別の要約）が使う丸め・範囲決定もここに置く。**
`observed_age` / `ObservedAges` / `observed_ages` はもともと参照平均の併記（FR-039）
専用に `domain.reference_baseline` へ置かれていたが、FR-039 の撤回（2026-09-03、
§0 項目 34）に伴い、FR-039 とは独立に使われているこの 3 つだけを移設した
（`application.demographic_observations` が §9.6.1 の要約のために読む）。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Final

from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel

#: モデルが対象とする年齢帯（§10.8）。`domain.reference_stats` にも同値の
#: `MIN_AGE`/`MAX_AGE` があるが、依存を作らずここでも独立に定義する
#: （§9.5 の参照統計セルと、この module の丸め・範囲決定は別の関心事のため）。
MIN_AGE: Final[int] = 18
MAX_AGE: Final[int] = 75


def build_demographic_estimate(
    *,
    estimated_age_years: float,
    gender_probabilities: tuple[float, ...],
    model_version: str,
    model_sha256: str,
    observed_at: datetime,
) -> DemographicEstimate:
    """モデルの生出力を検証する。対象外の年齢を切り詰めることはしない。"""
    age = float(estimated_age_years)
    probabilities = tuple(float(value) for value in gender_probabilities)
    if not isfinite(age) or not MIN_AGE <= age <= MAX_AGE:
        raise ValueError("estimated age is outside the supported 18-75 range")
    if len(probabilities) != 2 or any(not isfinite(value) for value in probabilities):
        raise ValueError("expected two finite gender probabilities")
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("gender probabilities must have a positive sum")
    normalized = (probabilities[0] / total, probabilities[1] / total)
    winning_index = 0 if normalized[0] >= normalized[1] else 1
    labels = (GenderLabel.FEMALE, GenderLabel.MALE)
    return DemographicEstimate(
        estimated_age_years=age,
        gender=labels[winning_index],
        gender_confidence=normalized[winning_index],
        gender_probabilities=normalized,
        model_version=model_version,
        model_sha256=model_sha256,
        observed_at=observed_at,
    )


def mean_gender_probabilities(
    probabilities: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """観測された (女性, 男性) 確率の要素ごと平均。空なら None。

    各 estimate の `gender_probabilities` は正規化済み（和 1）なので、平均も和 1 に近い。
    """
    if not probabilities:
        return None
    n = len(probabilities)
    female = sum(p[0] for p in probabilities) / n
    male = sum(p[1] for p in probabilities) / n
    return (female, male)


def observed_age(age_years: float) -> int | None:
    """参考年齢から、レポートの参考年齢要約（§9.6.1）が使う丸めた年齢（1 歳刻み）を決める。

    **丸めは画面と同じでなければならない。** HUD・吹き出しは参考年齢を
    `f"{age:.0f} 歳"`（偶数丸め）で出すため、ここで `int()` の切り捨てや
    `floor(x + 0.5)` を使うと、ちょうど .5 のときだけ画面の年齢とレポートの
    年齢がずれる。`round` は同じ偶数丸めである。

    18〜75（§10.8 と同じ範囲）の外と非有限値は `None`。境界へ clamp しない
    ——範囲外の顔に「18 歳」というラベルを当てることになるためである。
    """
    if not math.isfinite(age_years):
        return None
    age = round(age_years)
    if not MIN_AGE <= age <= MAX_AGE:
        return None
    return age


@dataclass(frozen=True, slots=True)
class ObservedAges:
    """§9.6.1: 1 セッションで観測した参考年齢の範囲。

    `low`〜`high` は**四分位範囲（真ん中 50%）**を 1 歳へ丸めて外側へ広げたもので、
    観測の min〜max ではない。1 フレームの外れ値（例: 1 回だけ 55 歳）で範囲が
    不当に広がるのを避けるためである。要約（FR-021）が「変動幅（真ん中 50%）」を
    使うのと同じ語・同じ理由にそろえてある。

    `median` はレポートへ出す代表値である。範囲だけを出すと「20〜28 歳」がどの
    あたりの顔に対するものか読み取れない。
    """

    median: int
    low: int
    high: int
    sample_count: int


def observed_ages(age_years: Sequence[float]) -> ObservedAges | None:
    """観測した参考年齢の列から、レポートの参考年齢要約（§9.6.1）に使う範囲を決める。

    各値は `observed_age` と**同じ丸め**を通す（画面の「27 歳」とレポートの年齢が
    食い違わないようにするため）。18〜75 の外と非有限値はそこで落ちる——clamp しない。
    落とした後に 1 件も残らなければ `None` を返す。

    四分位は `statistics.quantiles` の補間を使い、`low` は切り下げ、`high` は
    切り上げで整数へ寄せる。**外側へ広げる**のは、丸めで範囲が実際の観測より
    狭くなることを避けるためである。標本が 1 件のときは、その 1 歳がそのまま範囲になる。
    """
    ages = [age for age in (observed_age(value) for value in age_years) if age is not None]
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
