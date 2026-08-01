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
    parts = {
        "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=3),
        "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=3),
        "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=3),
        "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=3),
    }
    settings = {
        "metric_version": "symmetry-2d/test",
        "landmark_map_version": "test/1.0.0",
        "landmark_count": len(SYNTHETIC_FACE),
        "left_eye_corners": (0, 1),
        "right_eye_corners": (2, 3),
        "midline_indices": (4, 5, 6),
        "parts": parts,
        "sensitivity": dict.fromkeys(PART_NAMES, 30.0),
        "weights": dict.fromkeys(PART_NAMES, 0.25),
        "sensitivity_frozen": False,
    }
    settings.update(overrides)
    return build_metric_config(**settings)


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def test_a_perfectly_symmetric_face_scores_one_hundred() -> None:
    result = compute_symmetry(SYNTHETIC_FACE, _config())

    assert result.overall == pytest.approx(100.0, abs=1e-6)
    for name in PART_NAMES:
        assert result.part(name) == pytest.approx(100.0, abs=1e-6)


def test_perturbing_one_part_lowers_that_part_monotonically() -> None:
    scores = []
    for offset in (0.0, 1.0, 3.0, 8.0):
        points = SYNTHETIC_FACE.copy()
        points[19, 0] -= offset
        scores.append(compute_symmetry(points, _config()).part("mouth"))

    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


def test_perturbing_one_part_leaves_the_others_untouched() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 6.0

    result = compute_symmetry(points, _config())

    assert result.part("contour") == pytest.approx(100.0, abs=1e-6)
    assert result.part("nose") == pytest.approx(100.0, abs=1e-6)
    assert result.part("mouth") < 100.0


@pytest.mark.parametrize("degrees", [-28.0, -9.0, 11.0, 24.0])
def test_scores_are_invariant_to_roll(degrees: float) -> None:
    baseline = compute_symmetry(SYNTHETIC_FACE, _config()).overall

    rolled = compute_symmetry(SYNTHETIC_FACE @ _rotation(degrees).T, _config()).overall

    assert rolled == pytest.approx(baseline, abs=1e-6)


def test_scores_are_invariant_to_translation_and_scale() -> None:
    baseline = compute_symmetry(SYNTHETIC_FACE, _config()).overall

    moved = compute_symmetry(SYNTHETIC_FACE * 7.5 + np.array([300.0, -80.0]), _config()).overall

    assert moved == pytest.approx(baseline, abs=1e-6)


def test_scores_are_invariant_to_mirroring() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 5.0
    baseline = compute_symmetry(points, _config()).overall

    mirrored = points.copy()
    mirrored[:, 0] *= -1.0
    result = compute_symmetry(mirrored, _config()).overall

    assert result == pytest.approx(baseline, abs=1e-6)


def test_overall_is_the_configured_weighted_sum() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 4.0
    weights = {"contour": 0.4, "eyes": 0.3, "nose": 0.2, "mouth": 0.1}

    result = compute_symmetry(points, _config(weights=weights))

    expected = sum(weights[name] * (result.part(name) or 0.0) for name in PART_NAMES)
    assert result.overall == pytest.approx(expected, abs=1e-9)


def test_a_larger_sensitivity_penalises_the_same_deviation_more() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 4.0

    gentle = compute_symmetry(points, _config(sensitivity=dict.fromkeys(PART_NAMES, 10.0)))
    harsh = compute_symmetry(points, _config(sensitivity=dict.fromkeys(PART_NAMES, 90.0)))

    assert harsh.part("mouth") < gentle.part("mouth")


def test_scores_stay_inside_zero_to_one_hundred() -> None:
    points = SYNTHETIC_FACE.copy()
    points[19, 0] -= 5000.0

    result = compute_symmetry(points, _config())

    assert 0.0 <= result.part("mouth") <= 100.0
    assert 0.0 <= result.overall <= 100.0


def test_raw_distances_are_reported_alongside_the_scores() -> None:
    result = compute_symmetry(SYNTHETIC_FACE, _config())

    assert result.distance("mouth") == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_landmarks_produce_no_score(bad: float) -> None:
    points = SYNTHETIC_FACE.copy()
    points[13, 1] = bad

    with pytest.raises(SymmetryError):
        compute_symmetry(points, _config())


def test_zero_eye_distance_produces_no_score() -> None:
    points = SYNTHETIC_FACE.copy()
    points[[2, 3]] = points[[0, 1]]

    with pytest.raises(SymmetryError):
        compute_symmetry(points, _config())


def test_a_part_below_its_minimum_pair_count_is_missing_not_zero() -> None:
    """A short landmark set makes some pairs unusable; those parts go missing."""
    result = compute_symmetry(SYNTHETIC_FACE[:20], _config())

    assert result.part("mouth") is None
    assert result.part("contour") == pytest.approx(100.0, abs=1e-6)


def test_the_overall_score_is_missing_when_any_part_is_missing() -> None:
    result = compute_symmetry(SYNTHETIC_FACE[:20], _config())

    assert result.overall is None
    assert result.is_complete is False


def test_a_complete_result_converts_to_the_shared_score_model() -> None:
    score = compute_symmetry(SYNTHETIC_FACE, _config()).to_score()

    assert score is not None
    assert score.overall == pytest.approx(100.0, abs=1e-6)


def test_an_incomplete_result_does_not_convert_to_a_score() -> None:
    assert compute_symmetry(SYNTHETIC_FACE[:20], _config()).to_score() is None
