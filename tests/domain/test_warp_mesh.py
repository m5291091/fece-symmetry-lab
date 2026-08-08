import numpy as np
import pytest

from face_symmetry_lab.domain.warp_mesh import (
    BORDER_ANCHOR_COUNT,
    MAX_WARP_DISPLACEMENT_RATIO,
    WarpMeshError,
    border_anchors,
    build_warp_mesh,
    is_degenerate_triangle,
    is_orientation_flipped,
    signed_triangle_area,
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


# --- 外周アンカー ------------------------------------------------------------


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


# --- メッシュの組み立て ------------------------------------------------------


def test_the_mesh_appends_pinned_anchors_after_the_landmarks() -> None:
    """§8.6.4: 外周は両方の集合で同一なので、動きようがない。"""
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
    """端から 1 画素外にあるランドマークは副産物であって、失敗ではない。"""
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


# --- 変位ガード --------------------------------------------------------------


def test_a_displacement_inside_the_guard_is_accepted() -> None:
    target = _POINTS.copy()
    target[4, 0] += MAX_WARP_DISPLACEMENT_RATIO * 120.0 * 0.9

    assert _mesh(target).max_displacement_px > 0.0


def test_a_displacement_beyond_the_guard_refuses_the_whole_warp() -> None:
    """§8.6.4: これを超えると区分写像が自分自身へ折り重なるので、拒否する。"""
    target = _POINTS.copy()
    target[4, 0] += MAX_WARP_DISPLACEMENT_RATIO * 120.0 * 1.5

    with pytest.raises(WarpMeshError, match="変形量"):
        _mesh(target)


def test_the_guard_scales_with_the_face_rather_than_the_image() -> None:
    """同じ画素変位でも、大きな顔なら問題なく、小さな顔なら問題になる。"""
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


# --- 三角形の退化 ------------------------------------------------------------


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


# --- 三角形の向きの反転 ------------------------------------------------------

#: 線分 (70, 60)-(90, 60) をまたいで動く点。またぐ前後で三角形は裏返るが、
#: 面積の絶対値はどちらも 40px² のままである。
_ABOVE_THE_EDGE = np.array([[70.0, 60.0], [90.0, 60.0], [80.0, 56.0]])
_BELOW_THE_EDGE = np.array([[70.0, 60.0], [90.0, 60.0], [80.0, 64.0]])


def test_the_signed_area_keeps_the_winding_while_the_area_discards_it() -> None:
    triangle = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 6.0]])
    reversed_winding = triangle[::-1]

    assert signed_triangle_area(triangle) == pytest.approx(12.0)
    assert signed_triangle_area(reversed_winding) == pytest.approx(-12.0)
    assert triangle_area(reversed_winding) == pytest.approx(12.0)


def test_a_triangle_whose_winding_reverses_is_flipped() -> None:
    """§8.6.4: 裏返った三角形は逆向きのアフィン写像で画素を転送してしまう。"""
    assert is_orientation_flipped(_ABOVE_THE_EDGE, _BELOW_THE_EDGE) is True


def test_a_flip_is_caught_even_when_neither_triangle_is_a_sliver() -> None:
    """実測 421 件のうち 261 件は、両側とも 1px² を超える本物の反転だった。"""
    assert triangle_area(_ABOVE_THE_EDGE) > 1.0
    assert triangle_area(_BELOW_THE_EDGE) > 1.0
    assert is_degenerate_triangle(_ABOVE_THE_EDGE) is False
    assert is_degenerate_triangle(_BELOW_THE_EDGE) is False

    assert is_orientation_flipped(_ABOVE_THE_EDGE, _BELOW_THE_EDGE) is True


def test_a_matching_winding_is_not_flipped_however_small_the_triangle() -> None:
    """「面積が小さすぎる」と「向きが反転している」は別の破綻である。"""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.02]])
    target = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.01]])

    assert is_degenerate_triangle(source) is False  # 0.010px² ≥ 1e-3px²
    assert is_degenerate_triangle(target) is False  # 0.005px² ≥ 1e-3px²
    assert is_orientation_flipped(source, target) is False


def test_an_unmoved_triangle_is_never_flipped() -> None:
    assert is_orientation_flipped(_ABOVE_THE_EDGE, _ABOVE_THE_EDGE.copy()) is False


def test_the_displacement_guard_does_not_notice_a_flip() -> None:
    """§8.6.4: ガードは 1 点の *絶対* 変位、裏返りは隣接点の *相対* 変位で決まる。

    前者は後者を含意しない。この点は両眼間隔の 6.7% しか動いていないので
    ガードは働かないが、三角形は裏返っている。
    """
    source = np.vstack([_POINTS, _ABOVE_THE_EDGE])
    target = np.vstack([_POINTS, _BELOW_THE_EDGE])

    mesh = build_warp_mesh(source, target, width=WIDTH, height=HEIGHT, interocular_px=120.0)

    assert mesh.max_displacement_px == pytest.approx(8.0)
    assert mesh.max_displacement_px < MAX_WARP_DISPLACEMENT_RATIO * 120.0
    assert is_orientation_flipped(source[-3:], target[-3:]) is True


def test_a_wrongly_shaped_triangle_is_refused_by_the_orientation_check() -> None:
    with pytest.raises(WarpMeshError):
        is_orientation_flipped(np.zeros((4, 2)), np.zeros((4, 2)))
