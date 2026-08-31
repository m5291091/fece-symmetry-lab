"""FR-039 / §9.6.1: 1 セッションで**実際に表示できた**参考年齢・参考性別を溜める。

FR-042 の Camera レポートが参照平均を出すために要る。リアルタイムでは参考年齢が
数秒ごとに変わるため、Image のように 1 点では母集団を決められない。

**エクスポートの記録から作ってはならない。** 年齢ゲート（FR-038）が表示を止めた
フレームの推定も `SessionRecorder` には入っており、§9.2 の `age_gate_decision`
列がまだ実装されていないため、記録の側から「表示できた行」を見分けることもできない。
記録から母集団を作ると、**ゲートが隠した内部判断が母集団の年齢から復元できてしまう**
（§10.12-3、§10.11-2）。そこで、表示を許した瞬間にここへ溜める。

溜めるのは年齢と性別だけで、フレームも埋め込みも保持しない。上限を持つのは
`SessionRecorder` の履歴と同じ理由である（セッションが長くても有界に保つ）。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from face_symmetry_lab.domain.demographics import mean_gender_probabilities
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel
from face_symmetry_lab.domain.reference_baseline import ObservedAges, observed_ages

#: 溜める観測の上限。1Hz の推論で 30 分ぶんに相当する。四分位を取るのに十分で、
#: これを超えたセッションでは古いほうから捨てる（新しいほうが「いまの顔」に近い）。
DEFAULT_OBSERVATION_CAPACITY: int = 1_800


@dataclass(frozen=True, slots=True)
class DemographicOverview:
    """§9.6.1: 1 セッションで**実際に表示できた**参考年齢・参考性別の要約。

    FR-042 のレポートと Camera の専用タブが読む。1 点を持たないリアルタイムでは、
    「そのセッションが何歳・どの参考性別として表示されていたか」を範囲で述べる
    しかない。

    **年齢の丸めと範囲の決め方は参照平均の母集団と同じもの**（`observed_ages`）を
    使う。ここで 2 つ目の数え方を作ると、同じセッションのレポートの中で「31 歳」と
    「30 歳の平均」が並ぶ（§10.12-2）。

    **参考性別は観測フレームのモデル性別確率の平均**（女性・男性の確率）として持つ
    （2026-08-31 の指示）。判定が途中で入れ替わったセッションで 2 つのラベルが並ぶ
    と紛らわしいため、確率の平均に改めた。
    """

    ages: ObservedAges
    gender_probabilities: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DemographicObservationSet:
    """溜めた観測の不変ビュー。"""

    ages: tuple[float, ...]
    genders: tuple[GenderLabel, ...]
    gender_probabilities: tuple[tuple[float, float], ...]

    @property
    def is_empty(self) -> bool:
        return not self.ages


class DemographicObservations:
    """表示を許された推定だけを溜める、セッション単位の入れ物。

    `RealtimeLandmarkPipeline` が worker thread から書き、Streamlit の script
    thread が読むため、lock を持つ（`SessionRecorder` と同じ理由）。
    """

    def __init__(self, *, capacity: int = DEFAULT_OBSERVATION_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity は 1 以上である必要があります。")
        self._ages: deque[float] = deque(maxlen=capacity)
        self._genders: deque[GenderLabel] = deque(maxlen=capacity)
        self._gender_probabilities: deque[tuple[float, float]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def observe(self, estimate: DemographicEstimate | None, *, may_show: bool) -> None:
        """1 件の推定を、**表示を許されたときだけ**溜める。

        ``may_show`` は `AgeGateOutcome.may_show_reference_age`（ゲート未接続なら
        真）である。偽のときに溜めてしまうと、後からレポートの母集団としてゲートの
        判断が漏れる——**この 1 行がこの module の存在理由である**。
        """
        if estimate is None or not may_show:
            return
        with self._lock:
            self._ages.append(estimate.estimated_age_years)
            self._genders.append(estimate.gender)
            self._gender_probabilities.append(estimate.gender_probabilities)

    def snapshot(self) -> DemographicObservationSet:
        with self._lock:
            return DemographicObservationSet(
                ages=tuple(self._ages),
                genders=tuple(self._genders),
                gender_probabilities=tuple(self._gender_probabilities),
            )

    def reset(self) -> None:
        """§4.4: セッションのリセットで、溜めた観測も捨てる。"""
        with self._lock:
            self._ages.clear()
            self._genders.clear()
            self._gender_probabilities.clear()


def summarize_observations(
    observations: DemographicObservationSet,
) -> DemographicOverview | None:
    """観測から `DemographicOverview` を作る。表示できたものが無ければ `None`。

    **渡してよいのは `DemographicObservations.snapshot()` だけ**である——あちらは
    ゲートが表示を許した推定しか溜めない。記録から作った列を渡すと、ゲートが隠した
    判断がレポートの年齢から復元できてしまう（§10.12-3、§10.11-2）。

    年齢が 1 件も §9.5 の範囲（18〜75）に収まらないセッションでは `observed_ages`
    が `None` を返すので、ここも `None` になる——clamp して「18 歳」を作らない。
    参考性別は**観測フレームのモデル性別確率の平均**として持つ（2026-08-31 の指示）。
    判定が途中で入れ替わったセッションで 2 つのラベルが並ぶと紛らわしいため、
    確率の平均に改めた。
    """
    ages = observed_ages(observations.ages)
    if ages is None:
        return None
    probs = mean_gender_probabilities(observations.gender_probabilities)
    if probs is None:
        return None
    return DemographicOverview(ages=ages, gender_probabilities=probs)
