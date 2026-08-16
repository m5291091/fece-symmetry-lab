"""時間ベースの指数平滑化（§8.7）。

    alpha_t = 1 - exp(-delta_t / tau)
    ema_t   = alpha_t * raw_t + (1 - alpha_t) * ema_(t-1)

alpha を固定するとフレームレートが動くたびに時定数が変わってしまう。そのため
重みは、経過した単調時刻から導く。
"""

from __future__ import annotations

from math import exp, isfinite, log

#: §8.8: 解析がこの時間だけ滞ったら、フィルタをリセットする。
DEFAULT_MAX_GAP_SECONDS: float = 2.0
DEFAULT_TAU_SECONDS: float = 0.4

#: §15.2 が定める `ema_tau_seconds` の値域。FR-040 で利用者が選べるようになった
#: 今も、値域そのものは変わっていない。
MIN_TAU_SECONDS: float = 0.05
MAX_TAU_SECONDS: float = 5.0

#: FR-040: 画面に出す離散の段。連続スライダーにしない理由は ADR-0007 (c) にある
#: ——`tau` の効き方は対数的で、0.40 と 0.45 の差は画面上で判別できない。段に
#: すれば、選び直すたびに見た目が実際に変わる。既定（`DEFAULT_TAU_SECONDS`）は
#: この中に含まれていなければならない。含まれないと、起動直後の選択が「段の
#: どれでもない」状態になる。
TAU_CHOICES_SECONDS: tuple[float, ...] = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)

#: FR-040 が UI へ出す 2 つの追従割合。`RESPONSE_FRACTION_63` は
#: `1 - e⁻¹`（≒0.632）で、`response_time_seconds` へ渡すとちょうど `tau` を返す
#: ——この一致が「時定数とは何か」の説明そのものなので、UI 側で `0.63` と丸めた
#: 定数を書かず、ここから読む。
RESPONSE_FRACTION_63: float = 1.0 - exp(-1.0)
RESPONSE_FRACTION_95: float = 0.95


class SmoothingError(ValueError):
    """平滑化の入力または設定が不正である。"""


def validate_tau_seconds(value: float) -> float:
    """FR-040 の設定値を検証して返す。**clamp しない。**

    範囲外を境界へ丸めると、利用者の指定を黙って別の設定へ読み替えることになる。
    §15.2 の値域を外れた要求は、適用せずに拒否して直前の設定を保つ（呼び出し側の
    `application.realtime.RealtimeLandmarkPipeline.set_smoothing_tau_seconds` が
    そのように扱う）。
    """
    if not isfinite(value):
        raise SmoothingError("tau は有限値である必要があります。")
    if not MIN_TAU_SECONDS <= value <= MAX_TAU_SECONDS:
        raise SmoothingError(
            f"tau は {MIN_TAU_SECONDS}〜{MAX_TAU_SECONDS} 秒の範囲である必要があります。"
        )
    return float(value)


def nearest_tau_choice(tau_seconds: float | None) -> float:
    """``TAU_CHOICES_SECONDS`` のうち、与えられた値へ最も近い段を返す。

    段の外の `tau`（過去の設定、将来の既定変更、`FrameAnalyzer` を直接構築した
    テスト）で選択 UI が落ちないようにするための橋渡しである。`None`（平滑化を
    持たない pipeline）と非有限値は既定の段へ倒す。**ここは clamp してよい。**
    `validate_tau_seconds` が拒否するのは「利用者が指定した値」であるのに対し、
    こちらは「すでに動いている値を、段のどれとして見せるか」だけを決めている。
    """
    if tau_seconds is None or not isfinite(tau_seconds):
        return DEFAULT_TAU_SECONDS
    return min(TAU_CHOICES_SECONDS, key=lambda choice: abs(choice - tau_seconds))


def response_time_seconds(tau_seconds: float, *, fraction: float = 0.95) -> float:
    """表示が新しい水準へ ``fraction`` だけ近づくまでの時間（§8.7 の式の逆算）。

    `1 - exp(-t / tau) = fraction` を `t` について解いた `t = -tau × ln(1 - fraction)`
    であり、新しい定数を導入しない。`fraction=0.95` で `tau × ln 20`（tau の約 3.0 倍）、
    `fraction = 1 - e⁻¹`（≒0.632）でちょうど `tau` になる。

    `fraction` は開区間 `(0, 1)` に限る。1 は無限大、0 は「追従しない」で、どちらも
    表示する意味を持たない。
    """
    validate_tau_seconds(tau_seconds)
    if not isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise SmoothingError("追従割合は 0 より大きく 1 より小さい必要があります。")
    return -tau_seconds * log(1.0 - fraction)


def effective_sample_count(tau_seconds: float, *, interval_ms: int) -> float:
    """解析周期 ``interval_ms`` のもとで、いま何サンプル分を均しているかの目安。

    `tau ÷ 解析周期` であり、これも §8.7 の式から導かれるだけの量である。窓長を
    サンプル数で**指定させない**理由は ADR-0007 の alternatives にある——
    `analysis_interval_ms` は FR-030 の自動調整で動くため、同じ本数が指す実時間が
    黙って変わる。ここでは表示用の導出値としてだけ使う。
    """
    validate_tau_seconds(tau_seconds)
    if interval_ms <= 0:
        raise SmoothingError("解析周期は正の値である必要があります。")
    return tau_seconds / (interval_ms / 1000.0)


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
