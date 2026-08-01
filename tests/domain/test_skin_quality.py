import numpy as np
import pytest

from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.skin_quality import (
    SkinQualityError,
    compute_skin_quality,
    correct_camera_image,
)


def _frame() -> object:
    return build_landmark_frame(
        frame_id=1,
        face_count=1,
        points=((0.25, 0.35), (0.75, 0.35), (0.25, 0.65), (0.75, 0.65)),
        width=40,
        height=40,
    )


def test_gray_world_correction_reduces_channel_cast_and_records_gains() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[..., 0] = 180
    image[..., 1] = 100
    image[..., 2] = 60

    corrected, correction = correct_camera_image(image)

    before_spread = np.ptp(np.asarray(image, dtype=np.float64).mean(axis=(0, 1)))
    after_spread = np.ptp(np.asarray(corrected, dtype=np.float64).mean(axis=(0, 1)))
    assert after_spread < before_spread
    assert correction.gain_r < correction.gain_b
    assert all(0.5 <= gain <= 2.0 for gain in correction.gains)


def test_uniform_cheek_patches_have_a_high_reference_value() -> None:
    image = np.full((40, 40, 3), (150, 120, 105), dtype=np.uint8)

    result = compute_skin_quality(
        image,
        _frame(),  # type: ignore[arg-type]
        pairs=((0, 1), (2, 3)),
        patch_radius=3,
        min_pairs=2,
    )

    assert result.score == pytest.approx(100.0)
    assert result.tone_uniformity == pytest.approx(100.0)
    assert result.texture_uniformity == pytest.approx(100.0)
    assert result.frozen is False


def test_local_patch_variation_lowers_skin_reference_without_claiming_health() -> None:
    image = np.full((40, 40, 3), (150, 120, 105), dtype=np.uint8)
    rng = np.random.default_rng(7)
    image[11:18, 27:34] = rng.integers(20, 240, size=(7, 7, 3), dtype=np.uint8)

    result = compute_skin_quality(
        image,
        _frame(),  # type: ignore[arg-type]
        pairs=((0, 1), (2, 3)),
        patch_radius=3,
        min_pairs=2,
    )

    assert result.score is not None
    assert result.score < 100
    assert result.texture_uniformity is not None
    assert result.texture_uniformity < 100


def test_skin_correction_is_computed_from_the_face_region_not_background() -> None:
    frame = build_landmark_frame(
        frame_id=3,
        face_count=1,
        points=((0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)),
        width=60,
        height=60,
    )
    dark_background = np.full((60, 60, 3), (5, 5, 5), dtype=np.uint8)
    blue_background = np.full((60, 60, 3), (5, 5, 240), dtype=np.uint8)
    for image in (dark_background, blue_background):
        image[20:40, 20:40] = (160, 110, 80)

    dark_result = compute_skin_quality(
        dark_background,
        frame,
        pairs=((0, 1), (2, 3)),
        patch_radius=2,
    )
    blue_result = compute_skin_quality(
        blue_background,
        frame,
        pairs=((0, 1), (2, 3)),
        patch_radius=2,
    )

    assert dark_result.correction is not None
    assert blue_result.correction is not None
    assert blue_result.correction.gains == pytest.approx(dark_result.correction.gains)


def test_invalid_image_is_rejected_and_no_face_is_missing() -> None:
    with pytest.raises(SkinQualityError, match="RGB"):
        correct_camera_image(np.zeros((10, 10), dtype=np.uint8))

    empty = build_landmark_frame(frame_id=2, face_count=0)
    result = compute_skin_quality(
        np.zeros((20, 20, 3), dtype=np.uint8),
        empty,
        pairs=((0, 1),),
    )
    assert result.score is None
    assert result.pair_count == 0
