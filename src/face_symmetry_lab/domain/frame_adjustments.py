"""表示向けの、純粋なフレーム調整（§14.3b）。

ここにあるどちらの調整も採点の一部ではない。明るさ gain はランドマーク検出の
前にも適用する。暗い部屋では検出と閲覧の両方を助けうるためである。背景合成は
表示されるものにしか触れず、ランドマーク検出器や品質ゲートへ渡すフレームには
決して触れない。
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
    """§15.2 の Advanced 設定。表示専用であり、採点へは決して寄与しない。"""

    OFF = "OFF"
    BLACK = "BLACK"
    WHITE = "WHITE"


BACKGROUND_COLORS: Final[dict[BackgroundMode, tuple[int, int, int]]] = {
    BackgroundMode.BLACK: (0, 0, 0),
    BackgroundMode.WHITE: (255, 255, 255),
}


class FrameAdjustmentError(ValueError):
    """画像・mask・調整パラメータのいずれかが不正である。"""


def _validate_rgb(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    image = np.asarray(rgb_image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise FrameAdjustmentError("uint8のRGB画像が必要です。")
    return image


@lru_cache(maxsize=32)
def _brightness_lut(gain: float) -> NDArray[np.uint8]:
    """``gain`` に対する 256 要素の ``round(clip(value * gain, 0, 255))`` 表。

    キャッシュする理由は次のとおり。リアルタイム再生では
    `apply_brightness_gain` が *解析* フレームごとに 1 回呼ばれ（§14.3b: 表示
    出力は決して明るくしない）、その `gain` は自動調整（§14.3c）が更新したとき
    しか変わらない。その更新自体も解析周期よりずっと粗く間引かれている。毎回
    この表を作り直す処理は、全映像フレームがその費用を払っていた頃、gain が
    常に 1 以外である場合の支配的な費用だった。
    """
    return np.clip(np.rint(np.arange(256, dtype=np.float64) * gain), 0.0, 255.0).astype(np.uint8)


def apply_brightness_gain(rgb_image: NDArray[np.uint8], gain: float) -> NDArray[np.uint8]:
    """画素値へ ``gain`` を掛け、``[0, 255]`` へクリップして返す。"""
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
    """チャンネルごとの単純な画素平均。自動露出の信号として使う。

    知覚的な重み付け（ITU-R BT.601 など）は意図的に使わない。これは §14.3c の
    自動明るさ heuristic へ渡る値であり、そこで必要なのは gain を目標へ動かす
    ための安定したスカラーであって、測色的な測定値ではないためである。
    """
    image = _validate_rgb(rgb_image)
    return float(np.mean(image, dtype=np.float64))


def composite_background(
    rgb_image: NDArray[np.uint8],
    foreground_mask: NDArray[np.floating],
    background_color: tuple[int, int, int],
) -> NDArray[np.uint8]:
    """前景 mask を使って ``rgb_image`` と単色を合成する。

    ``foreground_mask`` は ``0``（背景）〜``1``（前景）で、画像と同じ高さ・幅を
    持つ。末尾のサイズ 1 のチャンネルは畳んで落とす。
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

    # §14.3b: 背景置換が有効な間、これは全映像フレームで走る（解析フレームは
    # 直接、間引かれたフレームはキャッシュ済み mask 経由）。そのためチャンネル
    # ごとに float32 で処理し、mask を (H, W) から (H, W, 3) へ broadcast する
    # のを避ける。その broadcast だけで素朴な float64 版は約 1.5〜2 倍遅かった。
    # 代償は最大 1 段階の丸め差だが、表示専用の合成では見分けが付かない。
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
