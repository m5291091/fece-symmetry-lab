"""撮影の明るさに対する非ブロッキング警告（§6.2, §6.3, §8.2）。

採点にも品質ゲートにも影響しない。生フレームの平均輝度だけを見て「明るさが
推奨範囲の外にある」ことを利用者へ伝える警告コードを返す。閾値は §8.2 のとおり
**暫定**で、実機評価で確定する。照度でスコアを止めない（ゲートへ昇格させない、
失敗分離）。顔領域ではなくフレーム全体の平均なので、明るい背景に暗い顔などは
取りこぼす——あくまで補助的な助言である。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.frame_adjustments import mean_luminance
from face_symmetry_lab.domain.models import ReasonCode

#: 0-255 の平均輝度がこれ未満なら LOW_LIGHT（暫定、§8.2）。
LOW_LIGHT_MEAN_MAX: float = 55.0
#: 0-255 の平均輝度がこれ超なら OVEREXPOSED（暫定、§8.2）。
OVEREXPOSED_MEAN_MIN: float = 215.0


def exposure_warnings(rgb_image: NDArray[np.uint8]) -> tuple[ReasonCode, ...]:
    """生フレームの平均輝度から非ブロッキングの照度警告を返す。範囲内なら空。

    `mean_luminance` は不正 shape で `FrameAdjustmentError` を送出する。ここでは
    握り潰さない——呼び出し側（`application.analyzer`）が失敗分離のため包む。
    """
    mean = mean_luminance(rgb_image)
    if mean < LOW_LIGHT_MEAN_MAX:
        return (ReasonCode.LOW_LIGHT,)
    if mean > OVEREXPOSED_MEAN_MIN:
        return (ReasonCode.OVEREXPOSED,)
    return ()
