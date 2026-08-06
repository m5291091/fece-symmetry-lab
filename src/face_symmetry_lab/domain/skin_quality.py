"""撮影条件で補正した、肌表面の参考指標。

ソフトウェア補正は単純な色かぶりを減らすだけであり、物理的なカメラ較正では
ない。出力は健康さや魅力を表すものでもない。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Final

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame

SKIN_METRIC_VERSION: Final[str] = "skin-surface/0.1.0"
MEDIAPIPE_CHEEK_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (50, 280),
    (101, 330),
    (117, 346),
    (123, 352),
    (205, 425),
)
MIN_GAIN: Final[float] = 0.5
MAX_GAIN: Final[float] = 2.0


class SkinQualityError(ValueError):
    """画像または頬 patch の設定を、安全に解析できない。"""


@dataclass(frozen=True, slots=True)
class CameraCorrection:
    gain_r: float
    gain_g: float
    gain_b: float
    luminance_low: float
    luminance_high: float

    @property
    def gains(self) -> tuple[float, float, float]:
        return (self.gain_r, self.gain_g, self.gain_b)


@dataclass(frozen=True, slots=True)
class SkinQualityResult:
    score: float | None
    tone_uniformity: float | None
    texture_uniformity: float | None
    pair_count: int
    correction: CameraCorrection | None
    metric_version: str = SKIN_METRIC_VERSION
    frozen: bool = False


def _validate_rgb(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise SkinQualityError("肌表面分析にはuint8のRGB画像が必要です。")
    return image


def _luminance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722


def _apply_correction(
    values: NDArray[np.float64],
    *,
    gains: NDArray[np.float64],
    low: float,
    high: float,
) -> NDArray[np.float64]:
    balanced = np.clip(values * gains, 0.0, 1.0)
    luminance = _luminance(balanced)
    if high - low >= 0.05:
        target_luminance = np.clip((luminance - low) / (high - low), 0.0, 1.0) * 0.8 + 0.1
        luminance_gain = np.clip(target_luminance / np.maximum(luminance, 1e-6), 0.5, 2.0)
        balanced = np.clip(balanced * luminance_gain[..., np.newaxis], 0.0, 1.0)
    return balanced


def correct_camera_image(
    rgb_image: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], CameraCorrection]:
    """範囲を制限した gray-world と、頑健な輝度正規化を局所的に適用する。"""
    image = _validate_rgb(rgb_image)
    values = np.asarray(image, dtype=np.float64) / 255.0
    channel_means = values.mean(axis=(0, 1))
    target_mean = float(channel_means.mean())
    gains = np.clip(target_mean / np.maximum(channel_means, 1e-6), MIN_GAIN, MAX_GAIN)
    initial = np.clip(values * gains, 0.0, 1.0)
    luminance = _luminance(initial)
    low, high = (float(value) for value in np.percentile(luminance, (5.0, 95.0)))
    balanced = _apply_correction(values, gains=gains, low=low, high=high)

    corrected = np.rint(balanced * 255.0).astype(np.uint8)
    correction = CameraCorrection(
        gain_r=float(gains[0]),
        gain_g=float(gains[1]),
        gain_b=float(gains[2]),
        luminance_low=low,
        luminance_high=high,
    )
    return corrected, correction


def _patch(
    image: NDArray[np.uint8],
    *,
    x: float,
    y: float,
    radius: int,
) -> NDArray[np.uint8] | None:
    height, width = image.shape[:2]
    center_x = min(max(round(x * (width - 1)), 0), width - 1)
    center_y = min(max(round(y * (height - 1)), 0), height - 1)
    left, right = center_x - radius, center_x + radius + 1
    top, bottom = center_y - radius, center_y + radius + 1
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    return image[top:bottom, left:right]


def _correct_face_region(
    image: NDArray[np.uint8],
    frame: LandmarkFrame,
) -> tuple[NDArray[np.uint8], CameraCorrection]:
    """ランドマークの外接矩形だけを補正し、背景色が支配しないようにする。"""
    height, width = image.shape[:2]
    xs = [point[0] for point in frame.points]
    ys = [point[1] for point in frame.points]
    left = max(0, int(np.floor(min(xs) * (width - 1))))
    right = min(width, int(np.ceil(max(xs) * (width - 1))) + 1)
    top = max(0, int(np.floor(min(ys) * (height - 1))))
    bottom = min(height, int(np.ceil(max(ys) * (height - 1))) + 1)
    if left >= right or top >= bottom:
        raise SkinQualityError("顔cropの範囲を決定できません。")
    _, correction = correct_camera_image(image[top:bottom, left:right])
    corrected_values = _apply_correction(
        np.asarray(image, dtype=np.float64) / 255.0,
        gains=np.asarray(correction.gains, dtype=np.float64),
        low=correction.luminance_low,
        high=correction.luminance_high,
    )
    corrected = np.rint(corrected_values * 255.0).astype(np.uint8)
    return corrected, correction


def compute_skin_quality(
    rgb_image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    pairs: tuple[tuple[int, int], ...],
    patch_radius: int = 3,
    min_pairs: int = 2,
) -> SkinQualityResult:
    """補正を前提とした、医学的でない頬表面の参考値を返す。"""
    image = _validate_rgb(rgb_image)
    if not frame.has_single_face or not frame.points:
        return SkinQualityResult(None, None, None, 0, None)
    if patch_radius < 1 or min_pairs < 1:
        raise SkinQualityError("patch_radiusとmin_pairsは1以上である必要があります。")
    count = len(frame.points)
    if any(left < 0 or right < 0 or left >= count or right >= count for left, right in pairs):
        raise SkinQualityError("landmark pairのindexが範囲外です。")

    corrected, correction = _correct_face_region(image, frame)
    texture_penalties: list[float] = []
    tone_penalties: list[float] = []
    patch_means: list[float] = []
    used_pairs = 0
    for left_index, right_index in pairs:
        pair_patches: list[NDArray[np.uint8]] = []
        for index in (left_index, right_index):
            point = frame.points[index]
            patch = _patch(corrected, x=point[0], y=point[1], radius=patch_radius)
            if patch is not None:
                pair_patches.append(patch)
        if len(pair_patches) != 2:
            continue
        used_pairs += 1
        for patch in pair_patches:
            values = np.asarray(patch, dtype=np.float64) / 255.0
            luminance = _luminance(values)
            gradient_y, gradient_x = np.gradient(luminance)
            texture_penalties.append(float(np.mean(np.hypot(gradient_x, gradient_y))))
            chroma = values - luminance[..., np.newaxis]
            tone_penalties.append(float(np.mean(np.std(chroma, axis=(0, 1)))))
            patch_means.append(float(luminance.mean()))

    if used_pairs < min_pairs:
        return SkinQualityResult(None, None, None, used_pairs, correction)
    texture_penalty = float(np.mean(texture_penalties))
    tone_penalty = float(np.mean(tone_penalties)) + float(np.std(patch_means))
    if not isfinite(texture_penalty) or not isfinite(tone_penalty):
        return SkinQualityResult(None, None, None, used_pairs, correction)
    texture_score = float(100.0 * exp(-8.0 * texture_penalty))
    tone_score = float(100.0 * exp(-6.0 * tone_penalty))
    score = 0.5 * texture_score + 0.5 * tone_score
    return SkinQualityResult(
        score=min(100.0, max(0.0, score)),
        tone_uniformity=min(100.0, max(0.0, tone_score)),
        texture_uniformity=min(100.0, max(0.0, texture_score)),
        pair_count=used_pairs,
        correction=correction,
    )
