import math

import numpy as np
import pytest

from face_symmetry_lab.domain.geometry import (
    fit_midline,
    normalization_transform,
    normalize_landmarks,
    reflect_across_midline,
)
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.symmetrization import (
    SymmetrizationError,
    symmetrize_landmarks,
)
from face_symmetry_lab.domain.symmetry import compute_symmetry
from tests.domain.test_symmetry import SYNTHETIC_FACE

#: Indices 0-24 are paired or midline; 25/26 stay out of every part below so
#: the "no correspondence defined" rule has something to leave alone.
UNPAIRED_INDICES = (25, 26)


def _config(**overrides):
    parts = {
        "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=2),
        "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=2),
        "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=2),
        "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=2),
    }
    settings = {
        "metric_version": "symmetry-2d/test",
        "landmark_map_version": "test/1.0.0",
        "landmark_count": len(SYNTHETIC_FACE),
        "left_eye_corners": (0, 1),
        "right_eye_corners": (2, 3),
        "midline_indices": (4, 5, 6),
        "parts": parts,
    }
    settings.update(overrides)
    return build_metric_config(**settings)


def _skewed(offset: float = 6.0):
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= offset
    points[7, 1] += offset / 2.0
    return points


# --- the transform's inverse -------------------------------------------------


def test_the_normalization_transform_round_trips_to_the_source_coordinates() -> None:
    """AC-SYMM-01: the inverse is exact, not an approximation."""
    transform = normalization_transform(
        SYNTHETIC_FACE, left_eye_corners=(0, 1), right_eye_corners=(2, 3)
    )

    restored = transform.invert(transform.apply(SYNTHETIC_FACE))

    assert np.allclose(restored, SYNTHETIC_FACE, atol=1e-6)


def test_the_transform_applies_the_same_normalization_as_before() -> None:
    transform = normalization_transform(
        SYNTHETIC_FACE, left_eye_corners=(0, 1), right_eye_corners=(2, 3)
    )

    assert np.allclose(
        transform.apply(SYNTHETIC_FACE),
        normalize_landmarks(SYNTHETIC_FACE, left_eye_corners=(0, 1), right_eye_corners=(2, 3)),
        atol=1e-12,
    )


def test_a_rolled_and_scaled_face_still_round_trips() -> None:
    angle = math.radians(23.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    points = SYNTHETIC_FACE @ rotation.T * 3.5 + np.array([120.0, -45.0])
    transform = normalization_transform(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))

    restored = transform.invert(transform.apply(points))

    assert np.allclose(restored, points, atol=1e-6)


# --- the symmetrized construction -------------------------------------------


def test_a_symmetric_face_is_already_its_own_symmetrization() -> None:
    result = symmetrize_landmarks(SYNTHETIC_FACE, _config())

    assert np.allclose(result.symmetrized, SYNTHETIC_FACE, atol=1e-6)
    assert result.max_deviation == pytest.approx(0.0, abs=1e-9)


def test_a_paired_point_moves_to_the_midpoint_of_itself_and_its_mirror() -> None:
    """§8.6.2's rule, checked against the geometry primitives it is built on."""
    points = _skewed()
    config = _config()
    normalized = normalize_landmarks(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    midline = fit_midline(normalized, config.midline_indices)
    mirrored = reflect_across_midline(normalized, midline)
    expected = (normalized[19] + mirrored[20]) / 2.0

    result = symmetrize_landmarks(points, config)

    transform = normalization_transform(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    assert np.allclose(transform.apply(result.symmetrized)[19], expected, atol=1e-9)


def test_the_right_side_is_the_mirror_of_the_symmetrized_left_side() -> None:
    points = _skewed()
    config = _config()
    result = symmetrize_landmarks(points, config)
    transform = normalization_transform(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    symmetrized = transform.apply(result.symmetrized)

    mirrored = reflect_across_midline(symmetrized, result.midline)

    for left, right in ((19, 20), (7, 8), (13, 14)):
        assert np.allclose(symmetrized[right], mirrored[left], atol=1e-9)


def test_a_midline_point_lands_on_the_fitted_line() -> None:
    """§8.6.2: `(p + mirror(p)) / 2` is the orthogonal projection."""
    points = _skewed()
    config = _config()
    result = symmetrize_landmarks(points, config)
    transform = normalization_transform(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    symmetrized = transform.apply(result.symmetrized)

    for index in config.midline_indices:
        relative = symmetrized[index] - result.midline.origin
        along = float(relative @ result.midline.direction)
        perpendicular = relative - along * result.midline.direction
        assert float(np.linalg.norm(perpendicular)) == pytest.approx(0.0, abs=1e-9)


def test_the_whole_symmetrized_set_is_symmetric_about_the_midline() -> None:
    points = _skewed()
    config = _config()
    result = symmetrize_landmarks(points, config)
    transform = normalization_transform(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    symmetrized = transform.apply(result.symmetrized)
    mirrored = reflect_across_midline(symmetrized, result.midline)

    for left, right in config.all_pairs:
        assert np.allclose(symmetrized[left], mirrored[right], atol=1e-9)


def test_points_without_a_defined_correspondence_keep_their_measured_position() -> None:
    """§8.6.2 refuses to guess a correspondence that the map does not define."""
    points = _skewed()

    result = symmetrize_landmarks(points, _config())

    for index in UNPAIRED_INDICES:
        assert index not in result.symmetrized_indices
        assert np.allclose(result.symmetrized[index], points[index], atol=1e-12)


def test_the_symmetrized_indices_cover_exactly_the_pairs_and_the_midline() -> None:
    config = _config()

    result = symmetrize_landmarks(_skewed(), config)

    expected = {index for pair in config.all_pairs for index in pair}
    expected.update(config.midline_indices)
    assert set(result.symmetrized_indices) == expected


def test_the_deviation_is_half_the_paired_distance_the_metric_reports() -> None:
    """§8.6.2 introduces no new metric: it is §8.5's distance, halved."""
    points = _skewed()
    config = _config()

    result = symmetrize_landmarks(points, config)
    normalized = normalize_landmarks(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))
    midline = fit_midline(normalized, config.midline_indices)
    mirrored = reflect_across_midline(normalized, midline)
    paired_distance = float(np.linalg.norm(normalized[19] - mirrored[20]))

    assert result.deviations[19] == pytest.approx(paired_distance / 2.0, abs=1e-9)
    assert result.deviations[20] == pytest.approx(paired_distance / 2.0, abs=1e-9)


def test_the_deviations_track_the_measured_distance_without_replacing_it() -> None:
    """A larger left/right difference shows up as a larger visual deviation."""
    small = symmetrize_landmarks(_skewed(2.0), _config())
    large = symmetrize_landmarks(_skewed(9.0), _config())

    assert small.max_deviation is not None and large.max_deviation is not None
    assert large.max_deviation > small.max_deviation
    # The metric it visualizes moves the same way, and stays the metric.
    small_overall = compute_symmetry(_skewed(2.0), _config()).overall
    large_overall = compute_symmetry(_skewed(9.0), _config()).overall
    assert small_overall is not None and large_overall is not None
    assert large_overall > small_overall


def test_the_result_carries_the_versions_needed_to_reproduce_it() -> None:
    result = symmetrize_landmarks(SYNTHETIC_FACE, _config())

    assert result.metric_version == "symmetry-2d/test"
    assert result.landmark_map_version == "test/1.0.0"


def test_a_short_landmark_set_symmetrizes_only_the_usable_pairs() -> None:
    result = symmetrize_landmarks(SYNTHETIC_FACE[:20], _config())

    assert 20 not in result.symmetrized_indices  # its partner is out of range
    assert 19 not in result.symmetrized_indices
    assert 7 in result.symmetrized_indices


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_landmarks_are_refused(bad: float) -> None:
    points = SYNTHETIC_FACE.copy()
    points[13, 1] = bad

    with pytest.raises(SymmetrizationError):
        symmetrize_landmarks(points, _config())


def test_a_zero_eye_distance_is_refused() -> None:
    points = SYNTHETIC_FACE.copy()
    points[[2, 3]] = points[[0, 1]]

    with pytest.raises(SymmetrizationError):
        symmetrize_landmarks(points, _config())


def test_the_input_points_are_never_mutated() -> None:
    points = _skewed()
    original = points.copy()

    symmetrize_landmarks(points, _config())

    assert np.array_equal(points, original)
