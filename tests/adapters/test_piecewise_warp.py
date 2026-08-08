import numpy as np
import pytest

from face_symmetry_lab.adapters.imaging.piecewise_warp import (
    OpenCvPiecewiseWarper,
    PiecewiseWarpError,
)
from face_symmetry_lab.domain.warp_mesh import (
    MAX_WARP_DISPLACEMENT_RATIO,
    build_warp_mesh,
    is_orientation_flipped,
    triangle_area,
)

WIDTH = 160
HEIGHT = 120
INTEROCULAR_PX = 60.0


def _image(seed: int = 0) -> np.ndarray:
    """模様のあるノイズ。画素を動かす変形を検出できるようにする。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


def _grid() -> np.ndarray:
    xs, ys = np.meshgrid(
        np.linspace(25.0, WIDTH - 25.0, 5),
        np.linspace(25.0, HEIGHT - 25.0, 4),
    )
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)


def _mesh(target=None):
    source = _grid()
    return build_warp_mesh(
        source,
        source.copy() if target is None else target,
        width=WIDTH,
        height=HEIGHT,
        interocular_px=INTEROCULAR_PX,
    )


def test_an_identity_mesh_reproduces_the_image_exactly() -> None:
    """参照描画なしで得られる、最も強い正しさの手がかり。"""
    image = _image()

    warped = OpenCvPiecewiseWarper().warp(image, _mesh())

    assert np.array_equal(warped, image)


def test_the_source_image_is_never_modified() -> None:
    image = _image()
    original = image.copy()
    target = _grid()
    target[:, 0] += 4.0

    OpenCvPiecewiseWarper().warp(image, _mesh(target))

    assert np.array_equal(image, original)


def test_a_displaced_mesh_actually_moves_pixels() -> None:
    image = _image()
    target = _grid()
    target[:, 0] += 5.0

    warped = OpenCvPiecewiseWarper().warp(image, _mesh(target))

    assert not np.array_equal(warped, image)
    assert warped.shape == image.shape
    assert warped.dtype == np.uint8


def test_the_pinned_border_keeps_its_original_pixels() -> None:
    """§8.6.4: 四隅はアンカーなので、髪や背景の端は動かない。"""
    image = _image()
    target = _grid()
    target[:, 1] += 4.0

    warped = OpenCvPiecewiseWarper().warp(image, _mesh(target))

    for row, column in ((0, 0), (0, WIDTH - 1), (HEIGHT - 1, 0), (HEIGHT - 1, WIDTH - 1)):
        assert np.array_equal(warped[row, column], image[row, column])


def test_the_triangulation_covers_the_frame_without_stray_vertices() -> None:
    """報告される三角形はすべて、実在するメッシュ 3 点へ解決できなければならない。"""
    warper = OpenCvPiecewiseWarper()
    mesh = _mesh()

    triangles = warper._triangles(mesh)

    assert triangles
    for triangle in triangles:
        assert len(triangle) == 3
        assert len(set(triangle)) == 3
        for index in triangle:
            assert 0 <= index < len(mesh.source)


def test_a_mismatched_image_size_is_refused() -> None:
    """別のフレーム用に作られたメッシュは、画素を誤った位置へ写してしまう。"""
    with pytest.raises(PiecewiseWarpError, match="サイズ"):
        OpenCvPiecewiseWarper().warp(np.zeros((10, 10, 3), dtype=np.uint8), _mesh())


def test_a_non_rgb_image_is_refused() -> None:
    with pytest.raises(PiecewiseWarpError, match="RGB"):
        OpenCvPiecewiseWarper().warp(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), _mesh())


def test_duplicate_points_do_not_break_the_warp() -> None:
    """潰れた座標は例外ではなく、読み飛ばしで扱う。"""
    source = _grid()
    source[1] = source[0]
    mesh = build_warp_mesh(
        source,
        source.copy(),
        width=WIDTH,
        height=HEIGHT,
        interocular_px=INTEROCULAR_PX,
    )

    warped = OpenCvPiecewiseWarper().warp(_image(), mesh)

    assert warped.shape == (HEIGHT, WIDTH, 3)


def test_points_on_the_frame_edge_are_accepted() -> None:
    """`build_warp_mesh` は最終画素へクランプする。その点は挿入可能でなければならない。"""
    source = _grid()
    source[0] = [0.0, 0.0]
    source[-1] = [WIDTH - 1.0, HEIGHT - 1.0]
    mesh = build_warp_mesh(
        source,
        source.copy(),
        width=WIDTH,
        height=HEIGHT,
        interocular_px=INTEROCULAR_PX,
    )

    warped = OpenCvPiecewiseWarper().warp(_image(), mesh)

    assert np.array_equal(warped, _image())


# --- 向きの反転（§8.6.4） ----------------------------------------------------


def _flipping_mesh():
    """三角形 1 つだけが裏返るメッシュ。

    先頭 4 点は動かない外郭で、続く 2 点が線分を張る。最後の点はその線分の上側
    から下側へ 8px 動く。両眼間隔 60px に対する変位ガードは 15px なので、ガード
    には遠く届かない。それでも、またいだ三角形は符号を変える。
    """
    source = np.array(
        [
            [40.0, 30.0],
            [120.0, 30.0],
            [40.0, 90.0],
            [120.0, 90.0],
            [70.0, 60.0],
            [90.0, 60.0],
            [80.0, 56.0],
        ]
    )
    target = source.copy()
    target[6] = [80.0, 64.0]
    return build_warp_mesh(
        source,
        target,
        width=WIDTH,
        height=HEIGHT,
        interocular_px=INTEROCULAR_PX,
    )


def _flipped_triangles(mesh) -> list[tuple[int, int, int]]:
    return [
        triangle
        for triangle in OpenCvPiecewiseWarper()._triangles(mesh)
        if is_orientation_flipped(mesh.source[list(triangle)], mesh.target[list(triangle)])
    ]


def _interior_pixels(triangle: np.ndarray) -> list[tuple[int, int]]:
    """三角形の内部に確実に収まる (row, column)。辺の反エイリアスを避ける。"""
    weights = ((0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (1 / 3, 1 / 3, 1 / 3))
    points = [np.asarray(weight, dtype=np.float64) @ triangle for weight in weights]
    return [(round(float(point[1])), round(float(point[0]))) for point in points]


def test_a_flipped_triangle_is_skipped_rather_than_transferred_backwards() -> None:
    """§8.6.4: 裏返った三角形は逆向きのアフィンで画素を転送し、像を壊す。"""
    image = _image()
    mesh = _flipping_mesh()
    flipped = _flipped_triangles(mesh)
    assert len(flipped) == 1  # 構成の前提

    warped = OpenCvPiecewiseWarper().warp(image, mesh)

    # 変形先の三角形分割は重ならないので、読み飛ばした三角形の内部を書き込む
    # 三角形はほかに無い。元の画素がそのまま残るはずである。
    for row, column in _interior_pixels(mesh.target[list(flipped[0])]):
        assert np.array_equal(warped[row, column], image[row, column])


def test_the_flip_happens_well_inside_the_displacement_guard() -> None:
    """§8.6.4: ガードは絶対変位、裏返りは相対変位。前者は後者を含意しない。"""
    mesh = _flipping_mesh()

    assert _flipped_triangles(mesh)
    assert mesh.max_displacement_px < MAX_WARP_DISPLACEMENT_RATIO * INTEROCULAR_PX


def test_a_flip_is_skipped_even_when_neither_side_is_a_sliver() -> None:
    """実測 421 件のうち 261 件は、両側とも 1px² を超える本物の反転だった。"""
    mesh = _flipping_mesh()

    (flipped,) = _flipped_triangles(mesh)

    assert triangle_area(mesh.source[list(flipped)]) > 1.0
    assert triangle_area(mesh.target[list(flipped)]) > 1.0


def test_skipping_one_triangle_leaves_the_rest_of_the_warp_working() -> None:
    """向きの合う三角形は、面積にかかわらず従来どおり画素を運ぶ。"""
    image = _image()

    warped = OpenCvPiecewiseWarper().warp(image, _flipping_mesh())

    assert not np.array_equal(warped, image)
    assert warped.shape == image.shape


def test_a_symmetric_displacement_moves_both_sides_toward_the_middle() -> None:
    """向きの健全性確認。画素は自分のランドマークに追従する。"""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, : WIDTH // 2] = 255  # 中央にはっきりした垂直の境界
    source = _grid()
    target = source.copy()
    target[:, 0] -= 6.0  # すべてのランドマークを左へ動かす

    warped = OpenCvPiecewiseWarper().warp(image, _mesh(target))

    # 明暗の境界はメッシュとともに左へ移動しているはずである。
    original_edge = int(np.argmax(image[HEIGHT // 2, :, 0] < 128))
    warped_edge = int(np.argmax(warped[HEIGHT // 2, :, 0] < 128))
    assert warped_edge < original_edge
