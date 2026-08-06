"""いつ何を自動調整するか（§14.3c, §14.3d, FR-029/030）。

`RealtimeLandmarkPipeline` から切り出した。「何かを変えるべきか」という方針
——移動窓と再調整の間引き——を、その答えを適用するフレーム pipeline から分離
するためである。このモジュールが決め、pipeline が実行する。

ここでの推奨はすべて表示専用か cadence 専用である。§8.6 が指標の感度係数と
部位の重みを取り除いたため、自動調整が到達できる採点影響のある設定はもはや
存在せず、ここから `config_revision`/区間/EMA のリセットへ至る経路も無い
（AC-AUTOTUNE-02）。

ここは landmarker・recorder・worker・analyzer のいずれにも触れない。どの
メソッドも推奨値か、「そのままにする」を意味する ``None`` を返すだけである。
これにより AC-AUTOTUNE-01/02 の失敗時規則を守りやすくなる。握り潰した例外が
「変更なし」と区別できない形になるからである。

設計上スレッドセーフではない。どのメソッドも `_analyse` から呼ばれ、そこは
`FrameScheduler` の容量 1 のゲートがすでに直列化している。その経路だけが触る
状態に対して lock は取らない。
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.auto_tuning import (
    recommend_analysis_interval_ms,
    recommend_brightness_gain,
    recommend_zoom_policy,
)
from face_symmetry_lab.domain.frame_adjustments import mean_luminance
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: §14.3c の明るさ・ズーム自動調整（FR-029）: 解析より遅い周期で計算し直す。
#: `apply_brightness_gain` は gain の値ごとに参照表をキャッシュするため、複数の
#: 解析フレームにわたって gain を安定させれば、毎回表を作り直さずにキャッシュを
#: 温かいまま保てる。
DISPLAY_RETUNE_INTERVAL_SECONDS: Final[float] = 1.0

#: §14.3d の cadence 自動調整（FR-030）: 採点に影響しないため区間のリセットは
#: 不要である。§8.6 が指標の係数を取り除いた結果、調整対象として残っているのは
#: cadence と表示だけであり、ここから測定値が動くことはない。
CADENCE_PROCESSING_WINDOW: Final[int] = 20
MIN_CADENCE_SAMPLES: Final[int] = 5
CADENCE_RETUNE_INTERVAL_SECONDS: Final[float] = 2.0


@dataclass(frozen=True, slots=True)
class DisplayTuning:
    """解析フレーム 1 枚から得た §14.3c の推奨。``None`` は現状維持を意味する。"""

    brightness_gain: float | None = None
    zoom_policy: TrackingZoomPolicy | None = None


#: 「何もしない」を表す共有インスタンス。通常経路で割り当てが起きないようにする。
_NO_DISPLAY_TUNING: Final[DisplayTuning] = DisplayTuning()


class _RetuneSchedule:
    """「もう時間か」を単調に判定し、成立時に自ら次の期限を進めるゲート。"""

    __slots__ = ("_interval_seconds", "_next_at")

    def __init__(self, interval_seconds: float, *, start_at: float) -> None:
        self._interval_seconds = interval_seconds
        self._next_at = start_at

    def due(self, at: float) -> bool:
        """1 間隔につき最大 1 回だけ True を返し、そのとき期限を進める。"""
        if at < self._next_at:
            return False
        self._next_at = at + self._interval_seconds
        return True


class AutoTuneController:
    """観測したフレーム統計から、§14.3c/d が変えたい対象を判断する。"""

    def __init__(self, *, enabled: bool, started_at: float) -> None:
        self._enabled = bool(enabled)
        self._display_schedule = _RetuneSchedule(
            DISPLAY_RETUNE_INTERVAL_SECONDS, start_at=started_at
        )
        self._cadence_schedule = _RetuneSchedule(
            CADENCE_RETUNE_INTERVAL_SECONDS, start_at=started_at
        )
        self._processing_ms_window: deque[float] = deque(maxlen=CADENCE_PROCESSING_WINDOW)

    @property
    def enabled(self) -> bool:
        """セッション中は固定である。§15.2 は実行時の切り替え手段を公開しない。"""
        return self._enabled

    def display_tuning(
        self,
        rgb_image: NDArray[np.uint8],
        frame: LandmarkFrame,
        *,
        at: float,
    ) -> DisplayTuning:
        """§14.3c: 明るさ gain とズーム方針。1 つの間引きを共有する。

        どちらも同じ瞬間の同じフレームから導かれるので、別々の周期へずれて
        いかないよう 1 つのスケジュールを共有する。`rgb_image` は補正前の生
        フレームでなければならない。§14.3b は表示出力を明るくしないため、その
        平均輝度は周囲光を直接読んだ値になり、推奨 gain は相対値ではなく絶対値
        になる。失敗は AC-AUTOTUNE-01 に従って握り潰し、値をそのままにする。
        """
        if not self._enabled or not self._display_schedule.due(at):
            return _NO_DISPLAY_TUNING

        try:
            brightness_gain: float | None = recommend_brightness_gain(mean_luminance(rgb_image))
        except Exception:
            brightness_gain = None

        zoom_policy: TrackingZoomPolicy | None = None
        # `face_area_ratio` は顔がちょうど 1 つ確認できた場合以外 None である。
        # それはズーム追跡が意味を持つ唯一の場合でもある。
        ratio = frame.face_area_ratio
        if ratio is not None:
            try:
                zoom_policy = recommend_zoom_policy(ratio)
            except Exception:
                zoom_policy = None
        return DisplayTuning(brightness_gain=brightness_gain, zoom_policy=zoom_policy)

    def cadence_tuning(
        self,
        processing_ms: float,
        *,
        at: float,
        current_interval_ms: int,
    ) -> int | None:
        """§14.3d: 新しい `analysis_interval_ms`。現状維持なら None を返す。

        推奨値がすでに有効な値と一致するときは None を返す。`age_interval_ms`
        と `history_capacity` はこの間隔の純粋な関数なので、ここで変化が無い
        ということはどちらも動かないことを意味する。省略することで、セッション
        中ずっと再調整のたびに recorder が履歴バッファを作り直すのを避けられる
        （AC-AUTOTUNE-02）。
        """
        if not self._enabled:
            return None
        # 標本は解析フレームごとに取り、*読む* のは間引かれた tick でだけである。
        self._processing_ms_window.append(processing_ms)
        if not self._cadence_schedule.due(at):
            return None
        if len(self._processing_ms_window) < MIN_CADENCE_SAMPLES:
            return None
        try:
            mean_processing_ms = statistics.fmean(self._processing_ms_window)
            recommended = recommend_analysis_interval_ms(mean_processing_ms)
        except Exception:
            return None
        return None if recommended == current_interval_ms else recommended
