"""Time-based exponential smoothing (§8.7).

    alpha_t = 1 - exp(-delta_t / tau)
    ema_t   = alpha_t * raw_t + (1 - alpha_t) * ema_(t-1)

A fixed alpha would change its time constant whenever the frame rate moves, so
the weight is derived from the elapsed monotonic time instead.
"""

from __future__ import annotations

from math import exp, isfinite

#: §8.8 resets the filter when analysis stalls for this long.
DEFAULT_MAX_GAP_SECONDS: float = 2.0
DEFAULT_TAU_SECONDS: float = 0.4


class SmoothingError(ValueError):
    """Invalid smoothing input or configuration."""


class TimeBasedEma:
    """Session-scoped smoother for the displayed overall score."""

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
        """The currently active time constant (§14.3d auto-tuning readback)."""
        return self._tau

    def reset(self) -> None:
        self._value = None
        self._at = None

    def update(self, raw: float, *, at: float) -> float:
        """Fold one raw score in and return the smoothed value."""
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
