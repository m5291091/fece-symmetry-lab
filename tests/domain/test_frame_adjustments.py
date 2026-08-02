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
    mask = np.full((4, 4), 5.0, dtype=np.float32)  # way above 1.0

    result = composite_background(image, mask, (0, 0, 0))

    assert np.array_equal(result, image)  # clipped to 1.0 -> fully foreground


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
