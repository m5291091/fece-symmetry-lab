"""Pure, display-oriented frame adjustments (§14.3b).

Neither adjustment here is part of scoring. Brightness gain is applied
before landmark detection too, since it can help both detection and
viewing in a dim room; background compositing only ever touches what is
displayed, never the frame handed to the landmark detector or the quality
gate.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray

MIN_BRIGHTNESS_GAIN: float = 0.25
MAX_BRIGHTNESS_GAIN: float = 3.0


class BackgroundMode(StrEnum):
    """§15.2 Advanced setting. Display-only; never feeds scoring."""

    OFF = "OFF"
    BLACK = "BLACK"
    WHITE = "WHITE"


BACKGROUND_COLORS: Final[dict[BackgroundMode, tuple[int, int, int]]] = {
    BackgroundMode.BLACK: (0, 0, 0),
    BackgroundMode.WHITE: (255, 255, 255),
}


class FrameAdjustmentError(ValueError):
    """Invalid image, mask, or adjustment parameter."""


def _validate_rgb(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise FrameAdjustmentError("uint8のRGB画像が必要です。")
    return image


@lru_cache(maxsize=32)
def _brightness_lut(gain: float) -> NDArray[np.uint8]:
    """The 256-entry ``round(clip(value * gain, 0, 255))`` table for ``gain``.

    Cached because realtime playback calls `apply_brightness_gain` once per
    *analysed* frame (§14.3b — display output is never brightened) with a
    `gain` that only changes when auto-tuning (§14.3c) updates it, which is
    itself throttled well below the analysis cadence. Building this table
    from scratch per call was the dominant cost of a continuously nonzero
    gain back when every video frame paid for it.
    """
    return np.clip(np.rint(np.arange(256, dtype=np.float64) * gain), 0.0, 255.0).astype(np.uint8)


def apply_brightness_gain(rgb_image: NDArray[np.uint8], gain: float) -> NDArray[np.uint8]:
    """Multiply pixel values by ``gain`` and clip back into ``[0, 255]``."""
    image = _validate_rgb(rgb_image)
    if not np.isfinite(gain) or not (MIN_BRIGHTNESS_GAIN <= gain <= MAX_BRIGHTNESS_GAIN):
        raise FrameAdjustmentError(
            f"brightness gain は {MIN_BRIGHTNESS_GAIN}〜{MAX_BRIGHTNESS_GAIN} "
            "の有限値である必要があります。"
        )
    if gain == 1.0:
        return image
    return _brightness_lut(gain)[image]


def mean_luminance(rgb_image: NDArray[np.uint8]) -> float:
    """The plain per-channel pixel average, used as an auto-exposure signal.

    Deliberately not a perceptual (e.g. ITU-R BT.601) weighting: this feeds
    §14.3c's auto brightness heuristic, which only needs a stable scalar to
    drive gain toward a target, not a colorimetric measurement.
    """
    image = _validate_rgb(rgb_image)
    return float(np.mean(image, dtype=np.float64))


def composite_background(
    rgb_image: NDArray[np.uint8],
    foreground_mask: NDArray[np.floating],
    background_color: tuple[int, int, int],
) -> NDArray[np.uint8]:
    """Blend ``rgb_image`` with a solid color using a foreground mask.

    ``foreground_mask`` is ``0`` (background) to ``1`` (foreground), same
    height/width as the image; a trailing size-1 channel is squeezed away.
    """
    image = _validate_rgb(rgb_image)
    mask = np.asarray(foreground_mask, dtype=np.float64)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise FrameAdjustmentError("mask は画像と同じ高さ・幅の2次元配列である必要があります。")
    if not np.all(np.isfinite(mask)):
        raise FrameAdjustmentError("mask は有限値である必要があります。")
    if len(background_color) != 3 or any(not 0 <= c <= 255 for c in background_color):
        raise FrameAdjustmentError("background_color は 0〜255 の RGB 3 要素である必要があります。")

    # §14.3b: this runs on every video frame while background replacement is
    # on (analysed frames directly, throttled ones via the cached mask), so
    # per-channel float32 avoids broadcasting the mask from (H, W) to
    # (H, W, 3) — that broadcast alone made the naive float64 version ~1.5-2x
    # slower — at the cost of an at-most-1-step rounding difference, which is
    # invisible in a display-only composite.
    clipped_mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    inverse_mask = 1.0 - clipped_mask
    image_f32 = image.astype(np.float32)
    blended = np.empty_like(image)
    for channel in range(image.shape[-1]):
        color_channel = float(background_color[channel])
        blended[..., channel] = np.rint(
            image_f32[..., channel] * clipped_mask + color_channel * inverse_mask
        ).astype(np.uint8)
    return blended
