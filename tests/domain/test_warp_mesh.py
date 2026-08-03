import numpy as np
import pytest

from face_symmetry_lab.domain.warp_mesh import (
    BORDER_ANCHOR_COUNT,
    MAX_WARP_DISPLACEMENT_RATIO,
    WarpMeshError,
    border_anchors,
    build_warp_mesh,
    is_degenerate_triangle,
    triangle_area,
)

WIDTH = 200
HEIGHT = 160

_POINTS = np.array(
    [[40.0, 40.0], [160.0, 40.0], [40.0, 120.0], [160.0, 120.0], [100.0, 80.0]],
    dtype=np.float64,
)


def _mesh(target=None, **overrides):
    settings = {
        "width": WIDTH,
        "height": HEIGHT,
        "interocular_px": 120.0,
    }
    settings.update(overrides)
    return build_warp_mesh(
        _POINTS,
        _POINTS.copy() if target is None else target,
        **settings,
    )


# --- border anchors ----------------------------------------------------------


def test_the_border_anchors_are_the_corners_and_edge_midpoints() -> None:
    anchors = border_anchors(WIDTH, HEIGHT)

    assert len(anchors) == BORDER_ANCHOR_COUNT
    corners = {(0.0, 0.0), (199.0, 0.0), (199.0, 159.0), (0.0, 159.0)}
    assert corners <= {(float(x), float(y)) for x, y in anchors}


def test_the_anchors_sit_inside_the_image() -> None:
    anchors = border_anchors(WIDTH, HEIGHT)

    assert anchors[:, 0].min() >= 0.0
    assert anchors[:, 1].min() >= 0.0
    assert anchors[:, 0].max() <= WIDTH - 1
    assert anchors[:, 1].max() <= HEIGHT - 1


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-4, 10)])
def test_a_degenerate_image_size_is_refused(width: int, height: int) -> None:
    with pytest.raises(WarpMeshError):
        border_anchors(width, height)


# --- mesh assembly -----------------------------------------------------------


def test_the_mesh_appends_pinned_anchors_after_the_landmarks() -> None:
    """§8.6.4: the border is identical in both sets, so it cannot move."""
    mesh = _mesh()

    assert mesh.landmark_count == len(_POINTS)
    assert len(mesh.source) == len(_POINTS) + BORDER_ANCHOR_COUNT
    assert mesh.anchor_indices == tuple(range(len(_POINTS), len(mesh.source)))
    for index in mesh.anchor_indices:
        assert np.array_equal(mesh.source[index], mesh.target[index])


def test_an_identity_target_moves_nothing() -> None:
    mesh = _mesh()

    assert mesh.max_displacement_px == pytest.approx(0.0)
    assert np.array_equal(mesh.source, mesh.target)


def test_the_mesh_reports_the_largest_displacement() -> None:
    target = _POINTS.copy()
    target[4, 0] += 3.0

    mesh = _mesh(target)

    assert mesh.max_displacement_px == pytest.approx(3.0)


def test_points_outside_the_frame_are_clamped_rather_than_refused() -> None:
    """A landmark a pixel outside the edge is an artefact, not a failure."""
    source = _POINTS.copy()
    source[0] = [-15.0, -22.0]
    target = source.copy()

    mesh = build_warp_mesh(source, target, width=WIDTH, height=HEIGHT, interocular_px=120.0)

    assert mesh.source[0].tolist() == [0.0, 0.0]
    assert mesh.source[:, 0].max() <= WIDTH - 1
    assert mesh.source[:, 1].max() <= HEIGHT - 1


def test_the_input_arrays_are_never_mutated() -> None:
    source = _POINTS.copy()
    source[0] = [-15.0, -22.0]
    original = source.copy()

    build_warp_mesh(source, source.copy(), width=WIDTH, height=HEIGHT, interocular_px=120.0)

    assert np.array_equal(source, original)


# --- the displacement guard --------------------------------------------------


def test_a_displacement_inside_the_guard_is_accepted() -> None:
    target = _POINTS.copy()
    target[4, 0] += MAX_WARP_DISPLACEMENT_RATIO * 120.0 * 0.9

    assert _mesh(target).max_displacement_px > 0.0


def test_a_displacement_beyond_the_guard_refuses_the_whole_warp() -> None:
    """§8.6.4: past this the piecewise map folds over itself, so refuse."""
    target = _POINTS.copy()
    target[4, 0] += MAX_WARP_DISPLACEMENT_RATIO * 120.0 * 1.5

    with pytest.raises(WarpMeshError, match="変形量"):
        _mesh(target)


def test_the_guard_scales_with_the_face_rather_than_the_image() -> None:
    """The same pixel displacement is fine on a big face and not on a small one."""
    target = _POINTS.copy()
    target[4, 0] += 20.0

    assert _mesh(target, interocular_px=120.0).max_displacement_px == pytest.approx(20.0)
    with pytest.raises(WarpMeshError):
        _mesh(target, interocular_px=40.0)


@pytest.mark.parametrize("interocular_px", [0.0, -1.0, float("nan"), float("inf")])
def test_an_invalid_interocular_distance_is_refused(interocular_px: float) -> None:
    with pytest.raises(WarpMeshError):
        _mesh(interocular_px=interocular_px)


def test_mismatched_point_counts_are_refused() -> None:
    with pytest.raises(WarpMeshError, match="点数"):
        build_warp_mesh(_POINTS, _POINTS[:3], width=WIDTH, height=HEIGHT, interocular_px=120.0)


def test_an_empty_point_set_is_refused() -> None:
    empty = np.empty((0, 2), dtype=np.float64)

    with pytest.raises(WarpMeshError):
        build_warp_mesh(empty, empty, width=WIDTH, height=HEIGHT, interocular_px=120.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_coordinates_are_refused(bad: float) -> None:
    target = _POINTS.copy()
    target[1, 0] = bad

    with pytest.raises(WarpMeshError, match="有限値"):
        _mesh(target)


def test_a_wrongly_shaped_array_is_refused() -> None:
    with pytest.raises(WarpMeshError, match=r"\(N, 2\)"):
        build_warp_mesh(
            np.zeros((4, 3)), np.zeros((4, 3)), width=WIDTH, height=HEIGHT, interocular_px=120.0
        )


# --- triangle degeneracy -----------------------------------------------------


def test_a_triangle_area_is_the_usual_half_cross_product() -> None:
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]])

    assert triangle_area(triangle) == pytest.approx(12.0)


def test_a_collinear_triangle_is_degenerate() -> None:
    collinear = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

    assert is_degenerate_triangle(collinear) is True


def test_a_normal_triangle_is_not_degenerate() -> None:
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]])

    assert is_degenerate_triangle(triangle) is False


def test_a_wrongly_shaped_triangle_is_refused() -> None:
    with pytest.raises(WarpMeshError):
        triangle_area(np.zeros((4, 2)))
