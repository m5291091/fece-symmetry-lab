"""表示専用の合成処理単体のテスト（§14.3b, §1.2(17), FR-013）。

明るさ補正が検出器の入力にしか届かないこと、背景置換の失敗が表示だけの問題に
留まること、間引いたフレームが TTL の内側でのみ直前の結果を再利用することを
確認する。
"""

import numpy as np
import pytest

from face_symmetry_lab.application.preview_compositor import PreviewCompositor
from face_symmetry_lab.domain.frame_adjustments import (
    MAX_BRIGHTNESS_GAIN,
    BackgroundMode,
    FrameAdjustmentError,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BlackoutSegmenter:
    """すべてを背景として報告し、合成が画素を目に見えて変えるようにする。"""

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def segment(self, rgb_image: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        self.calls += 1
        height, width = rgb_image.shape[:2]
        return np.zeros((height, width), dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class BrokenSegmenter:
    def __init__(self) -> None:
        self.closed = False

    def segment(self, rgb_image: np.ndarray, *, timestamp_ms: int) -> np.ndarray:
        raise RuntimeError("segmentation failure")

    def close(self) -> None:
        self.closed = True


def _compositor(clock: FakeClock, **overrides) -> PreviewCompositor:
    settings: dict = {"clock": clock}
    settings.update(overrides)
    return PreviewCompositor(**settings)


def _frame(face_count: int = 1):
    points = ((0.5, 0.5),) if face_count == 1 else ()
    return build_landmark_frame(frame_id=1, face_count=face_count, points=points)


def _image(value: int = 40) -> np.ndarray:
    return np.full((16, 16, 3), value, dtype=np.uint8)


# --- 明るさ補正（§14.3b） ------------------------------------------------------


def test_an_out_of_range_brightness_gain_is_rejected_at_construction() -> None:
    """使い捨ての 1 画素へ一度適用し、domain 側と同じ境界を再利用する。"""
    clock = FakeClock()
    with pytest.raises(FrameAdjustmentError):
        _compositor(clock, brightness_gain=MAX_BRIGHTNESS_GAIN + 1.0)


def test_a_unit_gain_hands_back_the_same_array() -> None:
    """補正が要らないフレームでコピーを払わない。"""
    clock = FakeClock()
    compositor = _compositor(clock, brightness_gain=1.0)
    image = _image()

    assert compositor.brighten_for_detection(image) is image


def test_a_raised_gain_brightens_only_the_detector_input() -> None:
    clock = FakeClock()
    compositor = _compositor(clock)
    compositor.set_brightness_gain(1.5)
    image = _image(value=40)

    brightened = compositor.brighten_for_detection(image)

    assert int(brightened.max()) > 40
    assert int(image.max()) == 40  # 表示に返る生フレームは変わらない
    assert compositor.brightness_gain == pytest.approx(1.5)


# --- 描画（FR-013） ------------------------------------------------------------


def test_painting_returns_the_source_when_there_is_nothing_to_draw() -> None:
    """呼び出し側が、変更していないフレームの再エンコードを省けるようにする。"""
    clock = FakeClock()
    compositor = _compositor(clock)
    image = _image()

    assert compositor.paint(image, _frame(face_count=0)) is image


def test_painting_draws_landmarks_onto_a_copy() -> None:
    clock = FakeClock()
    compositor = _compositor(clock)
    image = _image(value=0)

    painted = compositor.paint(image, _frame())

    assert painted is not image
    assert int(painted.sum()) > 0
    assert int(image.sum()) == 0


# --- 表示キャッシュ（§1.2(17)） ------------------------------------------------


def test_a_throttled_frame_redraws_the_last_geometry_within_the_ttl() -> None:
    clock = FakeClock()
    compositor = _compositor(clock)
    compositor.remember_geometry(_frame(), at=clock())

    redrawn = compositor.draw_cached(_image(value=0), ttl_seconds=0.25)

    assert int(redrawn.sum()) > 0


def test_an_expired_cache_draws_nothing_rather_than_stale_geometry() -> None:
    clock = FakeClock()
    compositor = _compositor(clock)
    compositor.remember_geometry(_frame(), at=clock())

    clock.advance(1.0)
    image = _image(value=0)

    assert compositor.draw_cached(image, ttl_seconds=0.25) is image


def test_a_frame_without_a_single_face_drops_the_cached_geometry() -> None:
    """顔が外れた後に、以前の幾何が復活してはならない。"""
    clock = FakeClock()
    compositor = _compositor(clock)
    compositor.remember_geometry(_frame(), at=clock())

    compositor.remember_geometry(_frame(face_count=0), at=clock())
    image = _image(value=0)

    assert compositor.draw_cached(image, ttl_seconds=0.25) is image


def test_clearing_the_caches_drops_the_geometry_at_once() -> None:
    clock = FakeClock()
    compositor = _compositor(clock)
    compositor.remember_geometry(_frame(), at=clock())

    compositor.clear_caches()
    image = _image(value=0)

    assert compositor.draw_cached(image, ttl_seconds=0.25) is image


# --- 背景置換（§14.3b, §14.3f） -----------------------------------------------


def test_background_replacement_is_skipped_while_it_is_off() -> None:
    clock = FakeClock()
    segmenter = BlackoutSegmenter()
    compositor = _compositor(clock, segmenter=segmenter)
    painted = _image()

    result, elapsed_ms = compositor.apply_background(_image(), painted, 0, at=clock())

    assert result is painted
    assert elapsed_ms == 0.0
    assert segmenter.calls == 0


def test_background_replacement_without_a_segmenter_is_a_no_op() -> None:
    clock = FakeClock()
    compositor = _compositor(clock, background_mode=BackgroundMode.BLACK)
    painted = _image()

    result, elapsed_ms = compositor.apply_background(_image(), painted, 0, at=clock())

    assert result is painted
    assert elapsed_ms == 0.0


def test_an_enabled_background_is_composited_onto_the_display_frame() -> None:
    clock = FakeClock()
    compositor = _compositor(
        clock,
        segmenter=BlackoutSegmenter(),
        background_mode=BackgroundMode.BLACK,
    )

    result, _ = compositor.apply_background(_image(200), _image(200), 0, at=clock())

    assert int(result.max()) == 0  # 完全に置き換わっている


def test_segmentation_runs_on_the_source_so_overlays_never_leak_into_the_mask() -> None:
    clock = FakeClock()
    segmenter = BlackoutSegmenter()
    compositor = _compositor(
        clock,
        segmenter=segmenter,
        background_mode=BackgroundMode.BLACK,
    )
    source = _image(value=10)

    compositor.apply_background(source, _image(value=250), 0, at=clock())

    assert segmenter.calls == 1


def test_a_segmentation_failure_is_display_only_and_returns_the_frame() -> None:
    """NFR-017: ここでの失敗を通信の失敗と取り違えてはならない。"""
    clock = FakeClock()
    compositor = _compositor(
        clock,
        segmenter=BrokenSegmenter(),
        background_mode=BackgroundMode.BLACK,
    )
    painted = _image()

    result, _ = compositor.apply_background(_image(), painted, 0, at=clock())

    assert result is painted


def test_a_throttled_frame_reuses_the_last_mask_within_the_ttl() -> None:
    clock = FakeClock()
    compositor = _compositor(
        clock,
        segmenter=BlackoutSegmenter(),
        background_mode=BackgroundMode.BLACK,
    )
    compositor.apply_background(_image(200), _image(200), 0, at=clock())

    redrawn = compositor.draw_cached(_image(200), ttl_seconds=0.25)

    assert int(redrawn.max()) == 0


def test_an_expired_mask_passes_the_frame_through_unchanged() -> None:
    clock = FakeClock()
    compositor = _compositor(
        clock,
        segmenter=BlackoutSegmenter(),
        background_mode=BackgroundMode.BLACK,
    )
    compositor.apply_background(_image(200), _image(200), 0, at=clock())

    clock.advance(1.0)
    redrawn = compositor.draw_cached(_image(200), ttl_seconds=0.25)

    assert int(redrawn.max()) == 200


def test_closing_releases_the_segmenter() -> None:
    clock = FakeClock()
    segmenter = BlackoutSegmenter()
    compositor = _compositor(clock, segmenter=segmenter)

    compositor.close()

    assert segmenter.closed is True
