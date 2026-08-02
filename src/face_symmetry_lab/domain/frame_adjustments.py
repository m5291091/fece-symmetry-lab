"""Pure, display-oriented frame adjustments (§14.3b).

Neither adjustment here is part of scoring. Brightness gain is applied
before landmark detection too, since it can help both detection and
viewing in a dim room; background compositing only ever touches what is
displayed, never the frame handed to the landmark detector or the quality
gate.
"""

from __future__ import annotations

from enum import StrEnum
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
    scaled = np.clip(image.astype(np.float64) * gain, 0.0, 255.0)
    return np.rint(scaled).astype(np.uint8)


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

    clipped_mask = np.clip(mask, 0.0, 1.0)[..., np.newaxis]
    color = np.asarray(background_color, dtype=np.float64)
    blended = image.astype(np.float64) * clipped_mask + color * (1.0 - clipped_mask)
    return np.rint(blended).astype(np.uint8)
