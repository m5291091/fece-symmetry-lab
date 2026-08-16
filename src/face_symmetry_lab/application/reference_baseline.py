"""FR-039 / §10.12: 参照平均の解決（use case）。

参考年齢・参考性別から母集団を 1 つ決め、§9.5 のセルを合併した要約を返す。計算は
`domain.reference_baseline` と `domain.reference_stats` がすべて持ち、ここが受け持つ
のは 3 つだけである。

* **併記できない条件をすべて「併記しない」へ倒す。** 推定が無い・年齢が 18〜75 の外・
  一致セルが 0 件・件数が §9.5 のしきい値未満・参照データが読めない——どれも例外では
  なく `None` を返す（不変条件 12、AC-BASELINE-03）。参照統計は左右差の表示に対して
  上乗せでしかないので、失敗したときに消えるのは併記だけでよい。
* **同じ母集団を二度絞り込まない。** 映像上の HUD は 5 Hz で描き直されるが、参考年齢と
  参考性別は数秒に 1 度しか変わらない。`(gender, age)` を鍵にした小さな memo を持つ。
* **しきい値を渡し直さない。** `ReferenceBrowser.query` が既定のしきい値で `combine`
  するので、ここで `min_count` を触らない（触れば §10.12-4 が崩れる）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from face_symmetry_lab.application.demographic_observations import (
    DemographicObservationSet,
)
from face_symmetry_lab.application.reference_browser import ReferenceBrowser
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel
from face_symmetry_lab.domain.reference_baseline import (
    ObservedAges,
    baseline_age,
    baseline_filter,
    observed_ages,
    session_baseline_filter,
)
from face_symmetry_lab.domain.reference_stats import MetricSummary


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    """1 つの母集団（同じ参考年齢・同じ参考性別・全人種）の要約。

    `age` と `gender` は画面へそのまま出す。母集団が何であるかを書かずに平均だけを
    出すと、何と比べているのかが読み取れない（§10.12-5）。
    """

    age: int
    gender: GenderLabel
    count: int
    metrics: Mapping[str, MetricSummary]


@dataclass(frozen=True, slots=True)
class SessionBaselineSummary:
    """FR-039/§9.6.1: 1 セッション分の母集団（範囲と、観測された性別の集合）。

    `BaselineSummary` が 1 点を指すのに対し、こちらは範囲を指す。画面へ出すのは
    範囲・性別・件数と、観測した参考年齢の中央値である——範囲だけでは「どのあたりの
    顔に対する平均か」が読み取れない（§10.12-5 と同じ理由）。
    """

    ages: ObservedAges
    genders: tuple[GenderLabel, ...]
    count: int
    metrics: Mapping[str, MetricSummary]


class BaselineProvider:
    """参照統計セルへの、併記のための唯一の窓口。

    `ReferenceBrowser` を組み立てるのは呼び出し側（`ui.baseline`）の仕事である。
    読み込みの失敗分離は browser 側が既に持っているので、ここでは重ねない。
    """

    def __init__(self, browser: ReferenceBrowser) -> None:
        self._browser = browser
        self._cache: dict[tuple[GenderLabel, int], BaselineSummary | None] = {}

    def resolve(self, estimate: DemographicEstimate | None) -> BaselineSummary | None:
        """参考年齢・参考性別に対応する母集団の要約。併記できないときは `None`。

        `estimate` が `None` のときに何も返さないことが、§10.12-3（参考年齢を
        表示していないときは母集団も出さない）の実装である。呼び出し側は
        「年齢ゲートが隠している推定」をここへ渡さないこと。
        """
        if estimate is None:
            return None
        age = baseline_age(estimate.estimated_age_years)
        if age is None:
            return None

        key = (estimate.gender, age)
        if key in self._cache:
            return self._cache[key]

        summary = self._query(age=age, gender=estimate.gender)
        self._cache[key] = summary
        return summary

    def _query(self, *, age: int, gender: GenderLabel) -> BaselineSummary | None:
        try:
            result = self._browser.query(baseline_filter(age=age, gender=gender))
        except Exception:  # 失敗分離。型を先読みしない（不変条件 12）。
            return None
        if result is None or result.aggregate is None:
            return None
        aggregate = result.aggregate
        if not aggregate.has_statistics:
            return None
        return BaselineSummary(
            age=age,
            gender=gender,
            count=aggregate.count,
            metrics=aggregate.metrics,
        )

    def resolve_session(
        self,
        observations: DemographicObservationSet,
    ) -> SessionBaselineSummary | None:
        """FR-039/§9.6.1: 1 セッション分の観測から母集団を 1 つ決める。

        `resolve` が 1 点（1 歳・1 性別）を指すのに対し、こちらは**四分位範囲と
        観測された性別の集合**を指す。リアルタイムでは参考年齢が数秒ごとに変わる
        ため、1 点では決められない。

        渡してよいのは `RealtimeLandmarkPipeline.demographic_observations`——
        **表示を許された推定だけ**——である。記録から作った列を渡してはならない
        （§10.12-3）。観測が無い・範囲が §9.5 の外・件数がしきい値未満・参照データが
        読めないのいずれでも `None` を返す（不変条件 12）。
        """
        if observations.is_empty:
            return None
        ages = observed_ages(observations.ages)
        if ages is None:
            return None
        genders = tuple(dict.fromkeys(observations.genders))
        if not genders:
            return None
        try:
            result = self._browser.query(session_baseline_filter(ages=ages, genders=genders))
        except Exception:  # 失敗分離。型を先読みしない（不変条件 12）。
            return None
        if result is None or result.aggregate is None:
            return None
        aggregate = result.aggregate
        if not aggregate.has_statistics:
            return None
        return SessionBaselineSummary(
            ages=ages,
            genders=genders,
            count=aggregate.count,
            metrics=aggregate.metrics,
        )
