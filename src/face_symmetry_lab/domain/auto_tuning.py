"""AI による Advanced 設定の自動調整のための純粋な heuristic（§14.3c/d, FR-029/030）。

新しいモデルは一切導入しない（不変条件 7, 10）。どの推奨も pipeline の他の場所
ですでに得られている統計——フレーム輝度、追跡中の顔 bbox 面積、実測の処理時間
——を再利用し、出力を固定範囲へクランプする。したがって自動調整された値が
FR-026/FR-028/FR-030 の範囲を超えることはない。

ここは採点に一切触れないし、触れられない。§8.6 が、採点に関わる唯一の設定で
あった感度係数と部位の重みを取り除いたためである。以下の推奨はすべて表示か
cadence の関心事である。品質ゲートの閾値も意図的に除外している。これを自動調整
すると、結果を測る妥当性の基準そのものを緩めてしまうからである（§14.3d）。
"""

from __future__ import annotations

import math
from typing import Final

from face_symmetry_lab.domain.frame_adjustments import MAX_BRIGHTNESS_GAIN, MIN_BRIGHTNESS_GAIN
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: 自動露出ループの中間グレー目標。一般的な 8bit カメラの「適正露出」の平均値
#: に合わせてある。
TARGET_MEAN_LUMINANCE: float = 128.0

#: 推奨方針が ``TrackingZoomPolicy()`` の既定値と一致する顔面積比。§8.2 の
#: face_area_min/max の中間領域に合わせて選んである。
_REFERENCE_FACE_AREA_RATIO: float = 0.12
_DEFAULT_POLICY = TrackingZoomPolicy()
_MIN_TARGET_PX = 240
_MAX_TARGET_PX = 960
_MIN_MARGIN_RATIO = 0.3
_MAX_MARGIN_RATIO = 1.2


class AutoTuningError(ValueError):
    """自動調整 heuristic へ渡されたフレーム統計が不正である。"""


def recommend_brightness_gain(measured_mean_luminance: float) -> float:
    """フレームを中間グレーへ寄せる絶対的な明るさ gain。

    ``measured_mean_luminance`` は常に補正前の生フレームの計測値である（§14.3b:
    gain はランドマーク検出器の入力にのみ適用され表示には及ばないため、「すでに
    補正された」フレームというものが存在しない）。したがって推奨値は
    ``target / measured`` そのものであり、FR-028 の
    ``[MIN_BRIGHTNESS_GAIN, MAX_BRIGHTNESS_GAIN]`` へクランプする。
    """
    if not math.isfinite(measured_mean_luminance) or measured_mean_luminance <= 0:
        raise AutoTuningError("measured_mean_luminance は正の有限値である必要があります。")
    gain = TARGET_MEAN_LUMINANCE / measured_mean_luminance
    return min(max(gain, MIN_BRIGHTNESS_GAIN), MAX_BRIGHTNESS_GAIN)


def recommend_zoom_policy(face_area_ratio: float) -> TrackingZoomPolicy:
    """顔が小さいほど強く寄る ``TrackingZoomPolicy`` を返す。

    PC の webcam はスマートフォンの自撮りよりはるかに小さく顔を写す（OD-05）。
    ここでは観測した顔面積と基準比の比の逆平方根で ``target_px``/``margin_ratio``
    を拡縮する。そのため遠い顔ほど切り出し目標と余白が大きくなり、近い顔ほど
    小さくなる。結果は ``TrackingZoomPolicy.__post_init__`` が FR-026 の範囲に
    対して再検証する。
    """
    if not math.isfinite(face_area_ratio) or not (0.0 < face_area_ratio <= 1.0):
        raise AutoTuningError("face_area_ratio は 0 超 1 以下の有限値である必要があります。")
    scale_factor = math.sqrt(_REFERENCE_FACE_AREA_RATIO / face_area_ratio)
    target_px = round(
        min(max(_DEFAULT_POLICY.target_px * scale_factor, _MIN_TARGET_PX), _MAX_TARGET_PX)
    )
    margin_ratio = min(
        max(_DEFAULT_POLICY.margin_ratio * scale_factor, _MIN_MARGIN_RATIO), _MAX_MARGIN_RATIO
    )
    return TrackingZoomPolicy(
        margin_ratio=margin_ratio,
        min_window_size=_DEFAULT_POLICY.min_window_size,
        recheck_interval_ms=_DEFAULT_POLICY.recheck_interval_ms,
        target_px=target_px,
        max_scale=_DEFAULT_POLICY.max_scale,
    )


#: §15.2 が文書化している候補そのもの。採点には一切関与しない（FR-030）。
ANALYSIS_INTERVAL_CHOICES_MS: Final[tuple[int, ...]] = (125, 200, 333)

#: 実測の処理時間を十分に上回る間隔のうち、最も速い候補を推奨する。解析ループ
#: が追随できる以上の負荷で回らないようにするためである。
_INTERVAL_UTILIZATION_TARGET: Final[float] = 0.7


def recommend_analysis_interval_ms(mean_processing_ms: float) -> int:
    """`ANALYSIS_INTERVAL_CHOICES_MS` のうち、系が維持できる最速の候補。"""
    if not math.isfinite(mean_processing_ms) or mean_processing_ms < 0:
        raise AutoTuningError("mean_processing_ms は 0 以上の有限値である必要があります。")
    for interval_ms in ANALYSIS_INTERVAL_CHOICES_MS:
        if mean_processing_ms <= interval_ms * _INTERVAL_UTILIZATION_TARGET:
            return interval_ms
    return ANALYSIS_INTERVAL_CHOICES_MS[-1]


MIN_AGE_INTERVAL_MS: Final[int] = 500
MAX_AGE_INTERVAL_MS: Final[int] = 10_000
#: 従来の固定既定値の比（解析 125ms / 年齢 1000ms）に一致させてある。
_AGE_INTERVAL_RATIO: Final[float] = 8.0


def recommend_age_interval_ms(analysis_interval_ms: int) -> int:
    """年齢・性別 worker の周期を、解析周期に比例させ続ける。"""
    if analysis_interval_ms <= 0:
        raise AutoTuningError("analysis_interval_ms は正の整数である必要があります。")
    recommended = round(analysis_interval_ms * _AGE_INTERVAL_RATIO)
    return min(max(recommended, MIN_AGE_INTERVAL_MS), MAX_AGE_INTERVAL_MS)


MIN_HISTORY_CAPACITY: Final[int] = 100
MAX_HISTORY_CAPACITY: Final[int] = 20_000
#: 周期によらず、バッファが覆うべき実時間の履歴長。
_HISTORY_TARGET_SECONDS: Final[float] = 600.0


def recommend_history_capacity(analysis_interval_ms: int) -> int:
    """与えられた周期で、一定の時間窓を覆うバッファ長。"""
    if analysis_interval_ms <= 0:
        raise AutoTuningError("analysis_interval_ms は正の整数である必要があります。")
    recommended = round(_HISTORY_TARGET_SECONDS * 1000.0 / analysis_interval_ms)
    return min(max(recommended, MIN_HISTORY_CAPACITY), MAX_HISTORY_CAPACITY)
