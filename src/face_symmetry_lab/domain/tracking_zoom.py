"""Pure geometry for realtime tracking-zoom (§14.3a, FR-026).

A PC webcam frames a face far smaller than a phone selfie does, so feeding
the raw frame straight to the landmark detector wastes most of its input
resolution on background. This module computes, in normalized [0, 1] frame
fractions only, the crop window to detect on next and how to map the
result back to full-frame coordinates. No image data or model calls happen
here, so it stays adapter-free and directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass


class TrackingZoomError(ValueError):
    """Invalid tracking-zoom geometry or configuration."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A face region expressed as fractions of the frame, in ``[0, 1]``."""

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


#: The whole frame, used for cold start, lost tracking, and periodic rechecks.
FULL_FRAME = BoundingBox(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class TrackingZoomPolicy:
    """§15.2 Advanced settings (currently internal defaults, UI unexposed)."""

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
    """Decide the crop window to detect on next.

    ``None`` means the previous frame did not settle on exactly one face
    (cold start, lost tracking, or a periodic multi-face recheck), so the
    caller must search the full frame again.
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
    """The tight bbox of a normalized point set, or ``None`` when degenerate."""
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
    """Map a point normalized to ``window`` back into full-frame fractions."""
    x, y = point
    return (window.x_min + x * window.width, window.y_min + y * window.height)


def remap_points(
    points: tuple[tuple[float, float], ...],
    window: BoundingBox,
) -> tuple[tuple[float, float], ...]:
    return tuple(remap_point(point, window) for point in points)


def resize_scale_for(
    crop_width_px: int,
    crop_height_px: int,
    *,
    policy: TrackingZoomPolicy | None = None,
) -> float:
    """A uniform upscale factor so a small crop reaches a workable resolution.

    Never downscales (returns at least 1.0), and never magnifies past
    ``policy.max_scale`` so a genuinely tiny, noisy face is not blown up into
    false precision.
    """
    if crop_width_px <= 0 or crop_height_px <= 0:
        raise TrackingZoomError("crop の pixel サイズは正の値である必要があります。")
    policy = policy or TrackingZoomPolicy()
    longest_side = max(crop_width_px, crop_height_px)
    scale = policy.target_px / longest_side
    return min(max(scale, 1.0), policy.max_scale)
