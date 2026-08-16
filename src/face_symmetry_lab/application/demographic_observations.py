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

from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel

#: 溜める観測の上限。1Hz の推論で 30 分ぶんに相当する。四分位を取るのに十分で、
#: これを超えたセッションでは古いほうから捨てる（新しいほうが「いまの顔」に近い）。
DEFAULT_OBSERVATION_CAPACITY: int = 1_800


@dataclass(frozen=True, slots=True)
class DemographicObservationSet:
    """溜めた観測の不変ビュー。"""

    ages: tuple[float, ...]
    genders: tuple[GenderLabel, ...]

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

    def snapshot(self) -> DemographicObservationSet:
        with self._lock:
            return DemographicObservationSet(
                ages=tuple(self._ages),
                genders=tuple(self._genders),
            )

    def reset(self) -> None:
        """§4.4: セッションのリセットで、溜めた観測も捨てる。"""
        with self._lock:
            self._ages.clear()
            self._genders.clear()
