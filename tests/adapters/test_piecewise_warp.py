import numpy as np
import pytest

from face_symmetry_lab.adapters.imaging.piecewise_warp import (
    OpenCvPiecewiseWarper,
    PiecewiseWarpError,
)
from face_symmetry_lab.domain.warp_mesh import build_warp_mesh

WIDTH = 160
HEIGHT = 120
INTEROCULAR_PX = 60.0


def _image(seed: int = 0) -> np.ndarray:
    """Textured noise, so a warp that moves pixels is detectable."""
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
    """The strongest correctness signal available without a reference render."""
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
    """§8.6.4: corners are anchors, so hair/background edges stay put."""
    image = _image()
    target = _grid()
    target[:, 1] += 4.0

    warped = OpenCvPiecewiseWarper().warp(image, _mesh(target))

    for row, column in ((0, 0), (0, WIDTH - 1), (HEIGHT - 1, 0), (HEIGHT - 1, WIDTH - 1)):
        assert np.array_equal(warped[row, column], image[row, column])


def test_the_triangulation_covers_the_frame_without_stray_vertices() -> None:
    """Every reported triangle must resolve to three real mesh points."""
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
    """A mesh built for another frame would map pixels to the wrong places."""
    with pytest.raises(PiecewiseWarpError, match="サイズ"):
        OpenCvPiecewiseWarper().warp(np.zeros((10, 10, 3), dtype=np.uint8), _mesh())


def test_a_non_rgb_image_is_refused() -> None:
    with pytest.raises(PiecewiseWarpError, match="RGB"):
        OpenCvPiecewiseWarper().warp(np.zeros((HEIGHT, WIDTH), dtype=np.uint8), _mesh())


def test_duplicate_points_do_not_break_the_warp() -> None:
    """Collapsed coordinates are skipped, not crashed on."""
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
    """`build_warp_mesh` clamps to the last pixel, which must stay insertable."""
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


def test_a_symmetric_displacement_moves_both_sides_toward_the_middle() -> None:
    """A sanity check on direction: pixels follow their landmarks."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, : WIDTH // 2] = 255  # a hard vertical edge at the centre
    source = _grid()
    target = source.copy()
    target[:, 0] -= 6.0  # drag every landmark left

    warped = OpenCvPiecewiseWarper().warp(image, _mesh(target))

    # The bright/dark boundary must have moved left with the mesh.
    original_edge = int(np.argmax(image[HEIGHT // 2, :, 0] < 128))
    warped_edge = int(np.argmax(warped[HEIGHT // 2, :, 0] < 128))
    assert warped_edge < original_edge
