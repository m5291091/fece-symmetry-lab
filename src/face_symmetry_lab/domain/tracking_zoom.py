"""リアルタイム追跡ズームのための純粋な幾何（§14.3a, FR-026）。

PC の webcam はスマートフォンの自撮りよりはるかに小さく顔を写すため、生の
フレームをそのままランドマーク検出器へ渡すと、入力解像度の大半を背景に費やす
ことになる。このモジュールは、正規化された [0, 1] のフレーム比だけを使って、
次に検出を行う切り出し窓と、その結果を全フレーム座標へ戻す方法を計算する。
画像データにも推論にも触れないので、adapter に依存せず直接テストできる。
"""

from __future__ import annotations

from dataclasses import dataclass


class TrackingZoomError(ValueError):
    """追跡ズームの幾何または設定が不正である。"""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """フレーム比 ``[0, 1]`` で表した顔の領域。"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x_min < self.x_max <= 1.0):
            raise TrackingZoomError("x_min < x_max が 0〜1 に収まる必要があります。")
        if not (0.0 <= self.y_min < self.y_max <= 1.0):
            raise TrackingZoomError("y_min < y_max が 0〜1 に収まる必要があります。")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


#: フレーム全体。初回・追跡喪失・定期的な再確認のときに使う。
FULL_FRAME = BoundingBox(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TrackingZoomPolicy:
    """§15.2 の Advanced 設定（現状は内部既定値で、UI へは公開しない）。"""

    margin_ratio: float = 0.6
    min_window_size: float = 0.18
    recheck_interval_ms: int = 1000
    target_px: int = 480
    max_scale: float = 4.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.margin_ratio <= 3.0:
            raise TrackingZoomError("margin_ratio は 0〜3.0 の範囲である必要があります。")
        if not 0.0 < self.min_window_size <= 1.0:
            raise TrackingZoomError("min_window_size は 0 より大きく 1 以下である必要があります。")
        if self.recheck_interval_ms < 0:
            raise TrackingZoomError("recheck_interval_ms は 0 以上である必要があります。")
        if self.target_px <= 0:
            raise TrackingZoomError("target_px は正の整数である必要があります。")
        if self.max_scale < 1.0:
            raise TrackingZoomError("max_scale は 1.0 以上である必要があります。")


def next_zoom_window(
    last_bbox: BoundingBox | None,
    *,
    policy: TrackingZoomPolicy | None = None,
) -> BoundingBox:
    """次に検出を行う切り出し窓を決める。

    ``None`` は、直前のフレームで顔がちょうど 1 つに定まらなかったこと（初回、
    追跡喪失、複数顔の定期再確認）を意味する。その場合、呼び出し側はフレーム
    全体を改めて探索しなければならない。
    """
    if last_bbox is None:
        return FULL_FRAME
    policy = policy or TrackingZoomPolicy()

    center_x = (last_bbox.x_min + last_bbox.x_max) / 2.0
    center_y = (last_bbox.y_min + last_bbox.y_max) / 2.0
    width = min(max(last_bbox.width * (1.0 + policy.margin_ratio), policy.min_window_size), 1.0)
    height = min(max(last_bbox.height * (1.0 + policy.margin_ratio), policy.min_window_size), 1.0)

    x_min = min(max(center_x - width / 2.0, 0.0), 1.0 - width)
    y_min = min(max(center_y - height / 2.0, 0.0), 1.0 - height)
    return BoundingBox(x_min, y_min, x_min + width, y_min + height)


def bounding_box_of(points: tuple[tuple[float, float], ...]) -> BoundingBox | None:
    """正規化された点集合の外接矩形。退化しているときは ``None``。"""
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_min, x_max = max(min(xs), 0.0), min(max(xs), 1.0)
    y_min, y_max = max(min(ys), 0.0), min(max(ys), 1.0)
    if x_min >= x_max or y_min >= y_max:
        return None
    return BoundingBox(x_min, y_min, x_max, y_max)


def remap_point(point: tuple[float, float], window: BoundingBox) -> tuple[float, float]:
    """``window`` を基準に正規化された点を、全フレーム比へ戻す。"""
    x, y = point
    return (window.x_min + x * window.width, window.y_min + y * window.height)


def remap_points(
    points: tuple[tuple[float, float], ...],
    window: BoundingBox,
) -> tuple[tuple[float, float], ...]:
    return tuple(remap_point(point, window) for point in points)


def remap_depths(
    depths: tuple[float, ...] | None,
    window: BoundingBox,
) -> tuple[float, ...] | None:
    """``window`` を基準に正規化された奥行きを、全フレーム比へ戻す（§8.1.1）。

    `z` は `x` と同じ係数 —— 画像の幅 —— で正規化されるので、写像は窓の *幅*
    比を掛けるだけである。xy と違って平行移動を持たないのは、`z` の原点が
    画像の端ではなく頭部中心付近にあるためである。切り出しの拡大率は切り出し
    画像と検出結果の双方へ等しく効くので、ここには現れない。

    奥行きが欠測（``None``）なら欠測のまま返す。0 で埋めない（不変条件 11）。
    """
    if depths is None:
        return None
    return tuple(depth * window.width for depth in depths)


def resize_scale_for(
    crop_width_px: int,
    crop_height_px: int,
    *,
    policy: TrackingZoomPolicy | None = None,
) -> float:
    """小さな切り出しが実用的な解像度へ届くための、一様な拡大率。

    縮小は決してしない（最低でも 1.0 を返す）。また ``policy.max_scale`` を
    超えて拡大もしない。本当に小さくノイズの多い顔を引き伸ばして、見かけ上の
    精度を作り出さないためである。
    """
    if crop_width_px <= 0 or crop_height_px <= 0:
        raise TrackingZoomError("crop の pixel サイズは正の値である必要があります。")
    policy = policy or TrackingZoomPolicy()
    longest_side = max(crop_width_px, crop_height_px)
    scale = policy.target_px / longest_side
    return min(max(scale, 1.0), policy.max_scale)
