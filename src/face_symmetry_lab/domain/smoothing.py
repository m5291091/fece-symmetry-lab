"""時間ベースの指数平滑化（§8.7）。

    alpha_t = 1 - exp(-delta_t / tau)
    ema_t   = alpha_t * raw_t + (1 - alpha_t) * ema_(t-1)

alpha を固定するとフレームレートが動くたびに時定数が変わってしまう。そのため
重みは、経過した単調時刻から導く。
"""

from __future__ import annotations

from math import exp, isfinite

#: §8.8: 解析がこの時間だけ滞ったら、フィルタをリセットする。
DEFAULT_MAX_GAP_SECONDS: float = 2.0
DEFAULT_TAU_SECONDS: float = 0.4


class SmoothingError(ValueError):
    """平滑化の入力または設定が不正である。"""


class TimeBasedEma:
    """表示する総合値のための、セッション単位の平滑化器。"""

    def __init__(
        self,
        *,
        tau_seconds: float = DEFAULT_TAU_SECONDS,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> None:
        if not isfinite(tau_seconds) or tau_seconds <= 0:
            raise SmoothingError("tau は正の有限値である必要があります。")
        if not isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise SmoothingError("max_gap は正の有限値である必要があります。")
        self._tau = tau_seconds
        self._max_gap = max_gap_seconds
        self._value: float | None = None
        self._at: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    @property
    def tau_seconds(self) -> float:
        """現在有効な時定数（§14.3d の自動調整の読み戻し）。"""
        return self._tau

    def reset(self) -> None:
        self._value = None
        self._at = None

    def update(self, raw: float, *, at: float) -> float:
        """生値を 1 つ取り込み、平滑化した値を返す。"""
        if not isfinite(raw):
            raise SmoothingError("平滑化の入力は有限値である必要があります。")
        if not isfinite(at):
            raise SmoothingError("平滑化の時刻は有限値である必要があります。")

        previous_at = self._at
        delta = None if previous_at is None else at - previous_at
        if self._value is None or delta is None or delta < 0.0 or delta >= self._max_gap:
            self._value = float(raw)
            self._at = at
            return self._value

        alpha = 1.0 - exp(-delta / self._tau) if delta > 0.0 else 0.0
        self._value = float(alpha * raw + (1.0 - alpha) * self._value)
        self._at = at
        return self._value
