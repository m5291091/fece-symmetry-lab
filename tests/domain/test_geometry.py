import math

import numpy as np
import pytest

from face_symmetry_lab.domain.geometry import (
    GeometryError,
    fit_midline,
    normalize_landmarks,
    reflect_across_midline,
)


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def _face_points() -> np.ndarray:
    """小さな合成顔。index 0/1 が左目の目頭・目尻、2/3 が右目の目頭・目尻。"""
    return np.array(
        [
            [-30.0, 0.0],
            [-10.0, 0.0],
            [10.0, 0.0],
            [30.0, 0.0],
            [0.0, 20.0],
            [-25.0, 40.0],
            [25.0, 40.0],
        ],
        dtype=np.float64,
    )


def _normalized(points: np.ndarray) -> np.ndarray:
    return normalize_landmarks(points, left_eye_corners=(0, 1), right_eye_corners=(2, 3))


def test_normalization_puts_the_eye_midpoint_at_the_origin() -> None:
    result = _normalized(_face_points())

    eye_midpoint = (result[[0, 1]].mean(axis=0) + result[[2, 3]].mean(axis=0)) / 2.0
    assert eye_midpoint == pytest.approx([0.0, 0.0], abs=1e-9)


def test_normalization_scales_the_eye_distance_to_one() -> None:
    result = _normalized(_face_points())

    left = result[[0, 1]].mean(axis=0)
    right = result[[2, 3]].mean(axis=0)
    assert float(np.linalg.norm(right - left)) == pytest.approx(1.0, abs=1e-9)


def test_normalization_levels_the_eye_line() -> None:
    rotated = _face_points() @ _rotation(23.0).T

    result = normalize_landmarks(rotated, left_eye_corners=(0, 1), right_eye_corners=(2, 3))

    left = result[[0, 1]].mean(axis=0)
    right = result[[2, 3]].mean(axis=0)
    assert float(right[1] - left[1]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("degrees", [-25.0, -7.5, 12.0, 30.0])
def test_normalization_is_invariant_to_roll(degrees: float) -> None:
    baseline = _normalized(_face_points())

    rolled = _normalized(_face_points() @ _rotation(degrees).T)

    assert rolled == pytest.approx(baseline, abs=1e-9)


@pytest.mark.parametrize("offset", [(120.0, -45.0), (-3.0, 900.0)])
def test_normalization_is_invariant_to_translation(offset: tuple[float, float]) -> None:
    baseline = _normalized(_face_points())

    shifted = _normalized(_face_points() + np.array(offset))

    assert shifted == pytest.approx(baseline, abs=1e-9)


@pytest.mark.parametrize("scale", [0.25, 3.0, 17.5])
def test_normalization_is_invariant_to_uniform_scale(scale: float) -> None:
    baseline = _normalized(_face_points())

    scaled = _normalized(_face_points() * scale)

    assert scaled == pytest.approx(baseline, abs=1e-9)


def test_zero_eye_distance_is_refused() -> None:
    points = _face_points()
    points[[2, 3]] = points[[0, 1]]

    with pytest.raises(GeometryError, match="眼"):
        _normalized(points)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_coordinates_are_refused(bad: float) -> None:
    points = _face_points()
    points[4, 0] = bad

    with pytest.raises(GeometryError, match="有限"):
        _normalized(points)


def test_eye_indices_outside_the_point_set_are_refused() -> None:
    with pytest.raises(GeometryError, match="範囲"):
        normalize_landmarks(_face_points(), left_eye_corners=(0, 99), right_eye_corners=(2, 3))


def test_midline_of_a_vertical_point_set_is_vertical() -> None:
    points = np.array([[0.0, -2.0], [0.0, 0.0], [0.0, 3.0], [0.0, 5.0]], dtype=np.float64)

    midline = fit_midline(points, (0, 1, 2, 3))

    assert abs(float(midline.direction[0])) == pytest.approx(0.0, abs=1e-9)
    assert abs(float(midline.direction[1])) == pytest.approx(1.0, abs=1e-9)


def test_midline_uses_orthogonal_distance_not_vertical_regression() -> None:
    """ほぼ水平な正中線でも、x = ay + b の回帰のように破綻してはならない。"""
    points = np.array([[-4.0, 0.0], [-1.0, 0.1], [2.0, -0.1], [5.0, 0.0]], dtype=np.float64)

    midline = fit_midline(points, (0, 1, 2, 3))

    assert abs(float(midline.direction[0])) > 0.9
    assert np.isfinite(midline.direction).all()


def test_midline_needs_three_distinct_points() -> None:
    points = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)

    with pytest.raises(GeometryError, match="正中"):
        fit_midline(points, (0, 1, 2))


def test_reflection_is_an_involution() -> None:
    points = _face_points()
    midline = fit_midline(np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]), (0, 1, 2))

    once = reflect_across_midline(points, midline)
    twice = reflect_across_midline(once, midline)

    assert twice == pytest.approx(points, abs=1e-9)


def test_reflection_across_a_vertical_midline_flips_x() -> None:
    points = np.array([[3.0, 1.0], [-2.0, -4.0]], dtype=np.float64)
    midline = fit_midline(np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]]), (0, 1, 2))

    reflected = reflect_across_midline(points, midline)

    expected = np.array([[-3.0, 1.0], [2.0, -4.0]], dtype=np.float64)
    assert reflected == pytest.approx(expected, abs=1e-9)


def test_reflection_preserves_distances() -> None:
    points = _face_points()
    midline = fit_midline(np.array([[1.0, -1.0], [1.2, 0.0], [1.4, 1.0]]), (0, 1, 2))

    reflected = reflect_across_midline(points, midline)

    original_span = np.linalg.norm(points[0] - points[6])
    reflected_span = np.linalg.norm(reflected[0] - reflected[6])
    assert float(reflected_span) == pytest.approx(float(original_span), abs=1e-9)
