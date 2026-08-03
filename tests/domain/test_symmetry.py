import math

import numpy as np
import pytest

from face_symmetry_lab.domain.metrics import PART_NAMES, PartMap, build_metric_config
from face_symmetry_lab.domain.symmetry import SymmetryError, compute_symmetry

# A synthetic face, exactly symmetric about x = 0.
SYNTHETIC_FACE = np.array(
    [
        [-30.0, 0.0],  # 0 left eye outer
        [-10.0, 0.0],  # 1 left eye inner
        [10.0, 0.0],  # 2 right eye inner
        [30.0, 0.0],  # 3 right eye outer
        [0.0, -20.0],  # 4 midline forehead
        [0.0, 20.0],  # 5 midline nose tip
        [0.0, 50.0],  # 6 midline chin
        [-40.0, 30.0],  # 7 contour
        [40.0, 30.0],  # 8
        [-45.0, 10.0],  # 9
        [45.0, 10.0],  # 10
        [-35.0, -10.0],  # 11
        [35.0, -10.0],  # 12
        [-8.0, 25.0],  # 13 nose
        [8.0, 25.0],  # 14
        [-12.0, 22.0],  # 15
        [12.0, 22.0],  # 16
        [-6.0, 28.0],  # 17
        [6.0, 28.0],  # 18
        [-15.0, 40.0],  # 19 mouth
        [15.0, 40.0],  # 20
        [-8.0, 38.0],  # 21
        [8.0, 38.0],  # 22
        [-5.0, 43.0],  # 23
        [5.0, 43.0],  # 24
        [-20.0, -5.0],  # 25 brow
        [20.0, -5.0],  # 26
    ],
    dtype=np.float64,
)


def _config(**overrides):
    # `min_pairs` deliberately sits below the pair count, as the real bundle's
    # does (30 of 63): the overall minimum is their sum, so the pooled value
    # keeps some headroom after a part has already gone missing.
    parts = {
        "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=2),
        "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=2),
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


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def test_a_perfectly_symmetric_face_has_zero_distance_everywhere() -> None:
    result = compute_symmetry(SYNTHETIC_FACE, _config())

    assert result.overall == pytest.approx(0.0, abs=1e-9)
    for name in PART_NAMES:
        assert result.distance(name) == pytest.approx(0.0, abs=1e-9)


def test_perturbing_one_part_raises_that_part_monotonically() -> None:
    distances = []
    for offset in (0.0, 1.0, 3.0, 8.0):
        points = SYNTHETIC_FACE.copy()
        points[19, 0] -= offset
        distances.append(compute_symmetry(points, _config()).distance("mouth"))

    assert distances == sorted(distances)
    assert distances[0] < distances[-1]


def test_perturbing_one_part_raises_the_overall_distance_monotonically() -> None:
    distances = []
    for offset in (0.0, 1.0, 3.0, 8.0):
        points = SYNTHETIC_FACE.copy()
        points[19, 0] -= offset
        distances.append(compute_symmetry(points, _config()).overall)

    assert distances == sorted(distances)
    assert distances[0] < distances[-1]


def test_perturbing_one_part_leaves_the_others_untouched() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 6.0

    result = compute_symmetry(points, _config())

    assert result.distance("contour") == pytest.approx(0.0, abs=1e-9)
    assert result.distance("nose") == pytest.approx(0.0, abs=1e-9)
    assert result.distance("mouth") > 0.0


@pytest.mark.parametrize("degrees", [-28.0, -9.0, 11.0, 24.0])
def test_distances_are_invariant_to_roll(degrees: float) -> None:
    baseline = compute_symmetry(SYNTHETIC_FACE, _config()).overall

    rolled = compute_symmetry(SYNTHETIC_FACE @ _rotation(degrees).T, _config()).overall

    assert rolled == pytest.approx(baseline, abs=1e-6)


def test_distances_are_invariant_to_translation_and_scale() -> None:
    """§8.3 normalizes the eye distance to 1, so a distance is a ratio, not pixels."""
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 4.0
    baseline = compute_symmetry(points, _config()).overall

    moved = compute_symmetry(points * 7.5 + np.array([300.0, -80.0]), _config()).overall

    assert moved == pytest.approx(baseline, abs=1e-6)


def test_distances_are_invariant_to_mirroring() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 5.0
    baseline = compute_symmetry(points, _config()).overall

    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    result = compute_symmetry(mirrored, _config()).overall

    assert result == pytest.approx(baseline, abs=1e-6)


def test_the_overall_distance_pools_every_pair_instead_of_weighting_parts() -> None:
    """§8.6: one fold over the pooled pairs, so each point counts exactly once."""
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 4.0
    points[7, 1] += 2.0
    config = _config()

    result = compute_symmetry(points, config)

    # Pooling is equivalent to the count-weighted quadratic mean, which the
    # equal-count parts here make an equal-weight quadratic mean. That is a
    # consequence of pooling, not a configurable weight.
    per_part = [result.distance(name) for name in PART_NAMES]
    expected = math.sqrt(sum(value**2 for value in per_part) / len(per_part))
    assert result.overall == pytest.approx(expected, abs=1e-9)
    # It is emphatically *not* the arithmetic mean the old Σ(w_p × score_p) used.
    assert result.overall != pytest.approx(sum(per_part) / len(per_part), abs=1e-9)


def test_pooling_weights_a_part_by_how_many_pairs_it_contributes() -> None:
    """A part with more pairs moves the pooled value more, by count alone."""
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 4.0
    wide_mouth = _config(
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=2),
            "nose": PartMap(pairs=((13, 14),), min_pairs=1),
            "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=3),
        }
    )
    narrow_mouth = _config(
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=3),
            "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=3),
            "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=3),
            "mouth": PartMap(pairs=((19, 20),), min_pairs=1),
        }
    )

    assert compute_symmetry(points, wide_mouth).overall > 0.0
    assert (
        compute_symmetry(points, wide_mouth).overall
        > compute_symmetry(points, narrow_mouth).overall
    )


def test_distances_are_non_negative_and_unbounded_above() -> None:
    """§8.6: 0 is the floor; there is no 100 to clamp against any more."""
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 20_000.0

    result = compute_symmetry(points, _config())

    # Far past the old 0-100 ceiling, which would have clamped this to 0.
    assert result.distance("mouth") > 100.0
    assert result.overall > 0.0
    assert math.isfinite(result.overall)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_landmarks_produce_no_distance(bad: float) -> None:
    points = SYNTHETIC_FACE.copy()
    points[13, 1] = bad

    with pytest.raises(SymmetryError):
        compute_symmetry(points, _config())


def test_zero_eye_distance_produces_no_distance() -> None:
    points = SYNTHETIC_FACE.copy()
    points[[2, 3]] = points[[0, 1]]

    with pytest.raises(SymmetryError):
        compute_symmetry(points, _config())


def test_a_part_below_its_minimum_pair_count_is_missing_not_zero() -> None:
    """A short landmark set makes some pairs unusable; those parts go missing."""
    result = compute_symmetry(SYNTHETIC_FACE[:20], _config())

    assert result.distance("mouth") is None
    assert result.distance("contour") == pytest.approx(0.0, abs=1e-9)


def test_a_missing_part_does_not_make_the_overall_distance_missing() -> None:
    """§8.6: the two are parallel outputs, so one is not composed of the other."""
    points = SYNTHETIC_FACE[:20].copy()
    points[7, 1] += 2.0

    result = compute_symmetry(points, _config())

    assert result.distance("mouth") is None
    assert result.overall is not None
    assert result.overall > 0.0
    assert result.is_complete is False


def test_the_overall_distance_is_missing_below_its_own_pair_minimum() -> None:
    """Too few usable pairs overall is missing, never zero (§8.6)."""
    result = compute_symmetry(SYNTHETIC_FACE[:8], _config())

    assert result.overall is None
    assert result.is_complete is False


def test_a_complete_result_converts_to_the_shared_distance_model() -> None:
    distance = compute_symmetry(SYNTHETIC_FACE, _config()).to_distance()

    assert distance is not None
    assert distance.overall == pytest.approx(0.0, abs=1e-9)


def test_an_incomplete_result_does_not_convert_to_a_distance() -> None:
    assert compute_symmetry(SYNTHETIC_FACE[:20], _config()).to_distance() is None
