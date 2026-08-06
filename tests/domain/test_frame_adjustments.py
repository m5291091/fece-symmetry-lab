import numpy as np
import pytest

from face_symmetry_lab.domain.frame_adjustments import (
    BACKGROUND_COLORS,
    MAX_BRIGHTNESS_GAIN,
    MIN_BRIGHTNESS_GAIN,
    BackgroundMode,
    FrameAdjustmentError,
    apply_brightness_gain,
    composite_background,
    mean_luminance,
)


def test_background_colors_are_defined_for_every_non_off_mode() -> None:
    assert BACKGROUND_COLORS[BackgroundMode.BLACK] == (0, 0, 0)
    assert BACKGROUND_COLORS[BackgroundMode.WHITE] == (255, 255, 255)
    assert BackgroundMode.OFF not in BACKGROUND_COLORS


def _image(height: int = 4, width: int = 4, value: int = 100) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_a_gain_of_one_returns_the_image_unchanged() -> None:
    image = _image(value=100)

    result = apply_brightness_gain(image, 1.0)

    assert np.array_equal(result, image)


def test_a_gain_above_one_brightens_and_clips_at_255() -> None:
    image = _image(value=200)

    result = apply_brightness_gain(image, 2.0)

    assert np.all(result == 255)


def test_a_gain_below_one_darkens_proportionally() -> None:
    image = _image(value=100)

    result = apply_brightness_gain(image, 0.5)

    assert np.all(result == 50)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf"), 0.1, 5.0])
def test_out_of_range_or_non_finite_gains_are_refused(gain: float) -> None:
    with pytest.raises(FrameAdjustmentError):
        apply_brightness_gain(_image(), gain)


def test_the_gain_bounds_are_inclusive() -> None:
    apply_brightness_gain(_image(), MIN_BRIGHTNESS_GAIN)
    apply_brightness_gain(_image(), MAX_BRIGHTNESS_GAIN)


def test_brightness_rejects_a_non_rgb_image() -> None:
    with pytest.raises(FrameAdjustmentError):
        apply_brightness_gain(np.zeros((4, 4), dtype=np.uint8), 1.5)


def test_a_full_foreground_mask_leaves_the_image_unchanged() -> None:
    image = _image(value=100)
    mask = np.ones((4, 4), dtype=np.float32)

    result = composite_background(image, mask, (0, 0, 0))

    assert np.array_equal(result, image)


def test_a_full_background_mask_returns_the_solid_color() -> None:
    image = _image(value=100)
    mask = np.zeros((4, 4), dtype=np.float32)

    result = composite_background(image, mask, (255, 255, 255))

    assert np.all(result == 255)


def test_a_partial_mask_blends_proportionally() -> None:
    image = _image(value=200)
    mask = np.full((4, 4), 0.5, dtype=np.float32)

    result = composite_background(image, mask, (0, 0, 0))

    assert np.all(result == 100)


def test_a_trailing_size_one_channel_on_the_mask_is_squeezed() -> None:
    image = _image(value=100)
    mask = np.ones((4, 4, 1), dtype=np.float32)

    result = composite_background(image, mask, (0, 0, 0))

    assert np.array_equal(result, image)


def test_mask_values_are_clipped_outside_zero_to_one() -> None:
    image = _image(value=100)
    mask = np.full((4, 4), 5.0, dtype=np.float32)  # 1.0 を大きく超える

    result = composite_background(image, mask, (0, 0, 0))

    assert np.array_equal(result, image)  # 1.0 へクリップされ、全面が前景になる


def test_a_mismatched_mask_shape_is_refused() -> None:
    with pytest.raises(FrameAdjustmentError):
        composite_background(_image(4, 4), np.ones((3, 3), dtype=np.float32), (0, 0, 0))


def test_a_non_finite_mask_is_refused() -> None:
    mask = np.ones((4, 4), dtype=np.float32)
    mask[0, 0] = float("nan")

    with pytest.raises(FrameAdjustmentError):
        composite_background(_image(4, 4), mask, (0, 0, 0))


@pytest.mark.parametrize("color", [(256, 0, 0), (-1, 0, 0), (0, 0)])
def test_an_invalid_background_color_is_refused(color: tuple) -> None:
    with pytest.raises(FrameAdjustmentError):
        composite_background(_image(4, 4), np.ones((4, 4), dtype=np.float32), color)


def test_composite_rejects_a_non_rgb_image() -> None:
    with pytest.raises(FrameAdjustmentError):
        composite_background(
            np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.float32), (0, 0, 0)
        )


@pytest.mark.parametrize("background_color", [(0, 0, 0), (255, 255, 255), (60, 120, 200)])
def test_composite_matches_the_reference_float64_blend_within_one_step(
    background_color: tuple[int, int, int],
) -> None:
    """内部実装が何であれ（遅い (H, W, 1) -> (H, W, 3) の broadcast を避ける
    ためのチャンネルごとの float32 ループなど）、出力は float64 の参照合成から
    丸め誤差の範囲に収まらねばならない。しかもきりの良い数値だけでなく、
    現実的できりの悪い mask・画素値についても成り立つ必要がある。"""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    mask = rng.random((64, 96)).astype(np.float32)

    result = composite_background(image, mask, background_color)

    clipped_mask = np.clip(mask.astype(np.float64), 0.0, 1.0)[..., np.newaxis]
    color = np.asarray(background_color, dtype=np.float64)
    reference = np.rint(
        image.astype(np.float64) * clipped_mask + color * (1.0 - clipped_mask)
    ).astype(np.uint8)
    assert np.abs(result.astype(int) - reference.astype(int)).max() <= 1


def test_mean_luminance_of_a_flat_image_is_its_pixel_value() -> None:
    assert mean_luminance(_image(value=64)) == pytest.approx(64.0)


def test_mean_luminance_averages_uneven_channels() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 30  # R
    image[..., 1] = 60  # G
    image[..., 2] = 90  # B

    assert mean_luminance(image) == pytest.approx(60.0)


def test_mean_luminance_rejects_a_non_rgb_image() -> None:
    with pytest.raises(FrameAdjustmentError):
        mean_luminance(np.zeros((4, 4), dtype=np.uint8))


@pytest.mark.parametrize("gain", [0.37, 0.99, 1.01, 1.3, 2.71, 2.99])
def test_gain_matches_the_reference_float64_transform_for_every_input_value(
    gain: float,
) -> None:
    """内部実装が何であれ（キャッシュした LUT など）、出力は float64 で計算した
    ``round(clip(pixel * gain, 0, 255))`` と、取りうるすべての uint8 入力値に
    ついて一致しなければならない。きりの良い数値だけでは足りない。"""
    gradient = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
    image = np.repeat(gradient, 3, axis=2)

    result = apply_brightness_gain(image, gain)

    reference = np.rint(np.clip(image.astype(np.float64) * gain, 0.0, 255.0)).astype(np.uint8)
    assert np.array_equal(result, reference)


def test_repeated_calls_with_different_gains_do_not_leak_into_each_other() -> None:
    """gain ごとにキャッシュした参照表を、別の gain へ使い回してはならない。"""
    image = _image(value=100)

    darker = apply_brightness_gain(image, 0.5)
    brighter = apply_brightness_gain(image, 2.0)
    darker_again = apply_brightness_gain(image, 0.5)

    assert np.all(darker == 50)
    assert np.all(brighter == 200)
    assert np.all(darker_again == 50)
