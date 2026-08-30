"""§6.2/§6.3/§8.2 の非ブロッキング照度警告の純粋計算テスト。

`domain.capture_quality.exposure_warnings` が生フレームの平均輝度から
`ReasonCode.LOW_LIGHT`/`OVEREXPOSED` を返すこと。採点・品質ゲートには
影響しない（非ブロッキック）。
"""

from __future__ import annotations

import numpy as np
import pytest

from face_symmetry_lab.domain.capture_quality import (
    LOW_LIGHT_MEAN_MAX,
    OVEREXPOSED_MEAN_MIN,
    exposure_warnings,
)
from face_symmetry_lab.domain.frame_adjustments import FrameAdjustmentError
from face_symmetry_lab.domain.models import ReasonCode


def _uniform_frame(value: int, height: int = 10, width: int = 10) -> np.ndarray:
    """全画素が `value` の RGB 画像を作る。"""
    return np.full((height, width, 3), value, dtype=np.uint8)


# --- 基本動作 -----------------------------------------------------------------


def test_dark_frame_returns_low_light() -> None:
    """暗いフレーム（mean < LOW_LIGHT_MEAN_MAX）→ (LOW_LIGHT,)。"""
    frame = _uniform_frame(20)
    assert exposure_warnings(frame) == (ReasonCode.LOW_LIGHT,)


def test_bright_frame_returns_overexposed() -> None:
    """明るいフレーム（mean > OVEREXPOSED_MEAN_MIN）→ (OVEREXPOSED,)。"""
    frame = _uniform_frame(230)
    assert exposure_warnings(frame) == (ReasonCode.OVEREXPOSED,)


def test_mid_range_frame_returns_empty() -> None:
    """中間（mean が [LOW_LIGHT_MEAN_MAX, OVEREXPOSED_MEAN_MIN] の範囲内）→ ()。"""
    frame = _uniform_frame(128)
    assert exposure_warnings(frame) == ()


# --- 閾値境界 -----------------------------------------------------------------


def test_exactly_at_low_light_threshold_is_empty() -> None:
    """平均値がちょうど LOW_LIGHT_MEAN_MAX（55.0）のときは empty（< でなく <=）。

    mean < LOW_LIGHT_MEAN_MAX が条件なので、ちょうど 55.0 は警告しない。
    """
    # 全画素 55（mean=55.0 = LOW_LIGHT_MEAN_MAX）
    value = int(LOW_LIGHT_MEAN_MAX)
    frame = _uniform_frame(value)
    # float mean = value.0 ≥ LOW_LIGHT_MEAN_MAX → empty
    assert exposure_warnings(frame) == ()


def test_one_below_low_light_threshold_triggers_warning() -> None:
    """平均値が LOW_LIGHT_MEAN_MAX より下 → LOW_LIGHT。"""
    value = int(LOW_LIGHT_MEAN_MAX) - 1  # 54
    frame = _uniform_frame(value)
    assert exposure_warnings(frame) == (ReasonCode.LOW_LIGHT,)


def test_exactly_at_overexposed_threshold_is_empty() -> None:
    """平均値がちょうど OVEREXPOSED_MEAN_MIN（215.0）のときは empty（> でなく >=）。

    mean > OVEREXPOSED_MEAN_MIN が条件なので、ちょうど 215.0 は警告しない。
    """
    value = int(OVEREXPOSED_MEAN_MIN)
    frame = _uniform_frame(value)
    # float mean = value.0 ≤ OVEREXPOSED_MEAN_MIN → empty
    assert exposure_warnings(frame) == ()


def test_one_above_overexposed_threshold_triggers_warning() -> None:
    """平均値が OVEREXPOSED_MEAN_MIN より上 → OVEREXPOSED。"""
    value = int(OVEREXPOSED_MEAN_MIN) + 1  # 216
    frame = _uniform_frame(value)
    assert exposure_warnings(frame) == (ReasonCode.OVEREXPOSED,)


# --- 罠検証 -------------------------------------------------------------------


def test_always_empty_implementation_fails_for_dark_frame() -> None:
    """罠: 常に () を返す実装が落ちることを確認。

    暗いフレームで非空タプルを要求する。
    """
    frame = _uniform_frame(20)  # mean=20 < 55
    result = exposure_warnings(frame)
    assert result != ()  # 常に () を返す実装はここで落ちる
    assert len(result) > 0


def test_always_empty_implementation_fails_for_bright_frame() -> None:
    """罠: 常に () を返す実装が落ちること（明るいフレーム）。"""
    frame = _uniform_frame(230)  # mean=230 > 215
    result = exposure_warnings(frame)
    assert result != ()
    assert len(result) > 0


def test_invalid_shape_propagates_frame_adjustment_error() -> None:
    """不正 shape（2D 配列）で FrameAdjustmentError が伝播する（握り潰さない）。"""
    bad_frame = np.zeros((10, 10), dtype=np.uint8)  # 2D: RGB ではない
    with pytest.raises(FrameAdjustmentError):
        exposure_warnings(bad_frame)


def test_invalid_dtype_propagates_frame_adjustment_error() -> None:
    """不正 dtype（float32）で FrameAdjustmentError が伝播する。"""
    bad_frame = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(FrameAdjustmentError):
        exposure_warnings(bad_frame)
