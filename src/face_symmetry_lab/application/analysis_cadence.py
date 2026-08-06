"""解析周期と、それに追随する表示キャッシュ TTL の管理。

`RealtimeLandmarkPipeline` から切り出した。「このフレームを解析するか」と
「解析しなかったフレームがどれだけの間、直前の結果を描き続けてよいか」は
同じ 1 つの周期から導かれるため、まとめて 1 箇所に置く。

自前の lock を持つ。周期の読み書きは WebRTC の受信スレッドと UI スレッドの
両方から起こるためである。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

#: プレビューのフレームは流れ続けるが、ランドマーク推論は 8 Hz に抑える。
DEFAULT_ANALYSIS_INTERVAL_MS = 125

#: FR-013: 間引かれたフレームが直前の幾何を描き直し、直前の背景 mask を
#: 再利用してよい時間。どちらも「解析しなかったフレームが何を表示してよいか」
#: という同じ問いに答えるため、1 つの TTL を共有する。
DEFAULT_DISPLAY_TTL_MS = 250

#: §1.2(17): 実効 TTL は下限を下回らないが、cadence の自動調整（§14.3d）が
#: 遅い周期を選んだときは `analysis_interval_ms * この値` まで広げる。
#: サンプリング間隔より短い TTL は毎周期キャッシュを失効させてしまい、それが
#: 333ms 候補で報告された背景置換のちらつきの原因だった。
_DISPLAY_TTL_SAFETY_MULTIPLIER = 2.0


class AnalysisCadence:
    """周期だけでフレームの採否を決める。間引いたフレームは後回しにも積まない。"""

    def __init__(
        self,
        *,
        interval_ms: int = DEFAULT_ANALYSIS_INTERVAL_MS,
        display_ttl_ms: int = DEFAULT_DISPLAY_TTL_MS,
        clock: Callable[[], float],
        started_at: float,
    ) -> None:
        if interval_ms < 0:
            raise ValueError("analysis_interval_ms は 0 以上である必要があります。")
        if display_ttl_ms < 0:
            raise ValueError("display_ttl_ms は 0 以上である必要があります。")
        self._clock = clock
        self._interval_seconds = interval_ms / 1000.0
        #: `widen_display_ttl` が広げる実効 TTL の下限。呼ぶのは cadence の
        #: 自動調整だけ（§14.3d）で、明示的に指定された
        #: `display_ttl_ms`/`analysis_interval_ms` の組（コンストラクタや
        #: `apply_advanced_settings`、テスト由来）はそのまま尊重する。
        self._base_display_ttl_seconds = display_ttl_ms / 1000.0
        self._display_ttl_seconds = self._base_display_ttl_seconds
        self._next_analysis_at = started_at
        self._throttled_frames = 0
        self._lock = threading.Lock()

    def admit(self) -> bool:
        """周期に照らして、このフレームを解析してよいかを返す。"""
        now = self._clock()
        with self._lock:
            if now < self._next_analysis_at:
                self._throttled_frames += 1
                return False
            self._next_analysis_at = now + self._interval_seconds
            return True

    @property
    def throttled_frames(self) -> int:
        """cadence 制御により表示専用のまま通したフレーム数。"""
        with self._lock:
            return self._throttled_frames

    @property
    def interval_ms(self) -> int:
        """現在有効な解析周期（§15.2、§14.3d の自動調整の読み戻し）。"""
        with self._lock:
            return round(self._interval_seconds * 1000)

    def set_interval_ms(self, interval_ms: int) -> None:
        """ランドマーク解析の周期を retune する（§15.2 `analysis_interval_ms`）。

        WebRTC のフレームレートからも `age_interval_ms` からも独立している。
        直後の間引き判定が古い周期を待たされないよう、即座に反映する。
        """
        if interval_ms < 0:
            raise ValueError("analysis_interval_ms は 0 以上である必要があります。")
        with self._lock:
            self._interval_seconds = interval_ms / 1000.0
            self._next_analysis_at = self._clock()

    @property
    def display_ttl_seconds(self) -> float:
        """現在有効な幾何・mask キャッシュの TTL（秒）。"""
        return self._display_ttl_seconds

    @property
    def display_ttl_ms(self) -> int:
        """同じ TTL の読み戻し（§1.2(17)）。

        cadence の自動調整（§14.3d）が常に `analysis_interval_ms` を上回るよう
        広げるため、解析済みフレームに挟まれた間引きフレームがキャッシュ切れに
        遭遇することはない。
        """
        return round(self._display_ttl_seconds * 1000)

    def widen_display_ttl(self) -> None:
        """§1.2(17): 自動調整された周期に追随して TTL を広げる。

        呼ぶのは cadence 自動調整の経路だけである。サンプリング間隔より短い
        TTL は毎周期キャッシュを失効させ、cadence 自動調整が 333ms 候補を
        選んだ際に報告された背景置換のちらつきを起こした。明示指定された
        `analysis_interval_ms`（コンストラクタや `apply_advanced_settings`）
        では呼ばず、呼び出し側が与えた `display_ttl_ms` をそのまま尊重する。
        """
        with self._lock:
            interval_seconds = self._interval_seconds
        self._display_ttl_seconds = max(
            self._base_display_ttl_seconds,
            interval_seconds * _DISPLAY_TTL_SAFETY_MULTIPLIER,
        )

    def reset(self) -> None:
        """次のフレームを直ちに解析対象にし、間引き数を 0 へ戻す。"""
        with self._lock:
            self._next_analysis_at = self._clock()
            self._throttled_frames = 0
