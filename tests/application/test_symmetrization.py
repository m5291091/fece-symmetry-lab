import numpy as np
import pytest

from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config
from face_symmetry_lab.domain.symmetrization import SymmetrizationError

WIDTH = 200
HEIGHT = 200

#: A symmetric face in normalized frame coordinates (see tests/domain).
_POINTS = (
    (0.35, 0.40),  # 0 left eye outer
    (0.45, 0.40),  # 1 left eye inner
    (0.55, 0.40),  # 2 right eye inner
    (0.65, 0.40),  # 3 right eye outer
    (0.50, 0.30),  # 4 midline
    (0.50, 0.50),  # 5 midline
    (0.50, 0.62),  # 6 midline
    (0.30, 0.55),  # 7 contour
    (0.70, 0.55),  # 8
    (0.31, 0.45),  # 9
    (0.69, 0.45),  # 10
    (0.33, 0.35),  # 11
    (0.67, 0.35),  # 12
    (0.47, 0.50),  # 13 nose
    (0.53, 0.50),  # 14
    (0.44, 0.58),  # 15 mouth
    (0.56, 0.58),  # 16
)


def _config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=2),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=2),
            "nose": PartMap(pairs=((13, 14),), min_pairs=1),
            "mouth": PartMap(pairs=((15, 16),), min_pairs=1),
        },
    )


def _frame(points=_POINTS, *, face_count: int = 1, width: int = WIDTH, height: int = HEIGHT):
    return build_landmark_frame(
        frame_id=1,
        face_count=face_count,
        points=points if face_count == 1 else (),
        width=width,
        height=height,
    )


def _image():
    return np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)


def _skewed():
    points = list(_POINTS)
    points[15] = (0.36, 0.58)
    return tuple(points)


def test_a_symmetric_face_produces_a_view_with_no_deviation() -> None:
    view = symmetrize_frame(_image(), _frame(), config=_config())

    assert view.max_deviation == pytest.approx(0.0, abs=1e-9)
    assert view.symmetrized_point_count == len(view.result.symmetrized_indices)
    assert view.overlay.shape == (HEIGHT, WIDTH, 3)


def test_a_skewed_face_reports_a_deviation_and_paints_the_overlay() -> None:
    image = _image()

    view = symmetrize_frame(image, _frame(_skewed()), config=_config())

    assert view.max_deviation is not None and view.max_deviation > 0.0
    assert view.mean_deviation is not None and view.mean_deviation > 0.0
    assert not np.array_equal(view.overlay, image)


def test_the_source_image_is_never_modified() -> None:
    """§14.3g is a read-only view over the captured frame."""
    image = _image()
    original = image.copy()

    symmetrize_frame(image, _frame(_skewed()), config=_config())

    assert np.array_equal(image, original)


def test_the_call_is_stateless_and_repeatable() -> None:
    """No EMA, no history: the same input always gives the same answer."""
    frame = _frame(_skewed())
    config = _config()

    first = symmetrize_frame(_image(), frame, config=config)
    second = symmetrize_frame(_image(), frame, config=config)

    assert first.max_deviation == pytest.approx(second.max_deviation)
    assert np.array_equal(first.overlay, second.overlay)


def test_the_symmetrized_positions_are_in_pixel_coordinates() -> None:
    view = symmetrize_frame(_image(), _frame(_skewed()), config=_config())

    measured = view.result.measured
    assert measured[:, 0].max() <= WIDTH
    assert measured[:, 1].max() <= HEIGHT
    # A near-symmetric face symmetrizes to positions inside the same frame.
    symmetrized = view.result.symmetrized[list(view.result.symmetrized_indices)]
    assert symmetrized[:, 0].min() >= 0.0
    assert symmetrized[:, 1].min() >= 0.0


def test_a_multi_face_frame_is_refused() -> None:
    """FR-008: one midline cannot stand for two faces."""
    with pytest.raises(SymmetrizationError, match="単一顔"):
        symmetrize_frame(_image(), _frame(face_count=2), config=_config())


def test_a_frame_without_image_dimensions_is_refused() -> None:
    with pytest.raises(SymmetrizationError):
        symmetrize_frame(_image(), _frame(width=0, height=0), config=_config())


def test_a_degenerate_face_is_refused_rather_than_drawn() -> None:
    collapsed = list(_POINTS)
    collapsed[2] = collapsed[0]
    collapsed[3] = collapsed[1]

    with pytest.raises(SymmetrizationError):
        symmetrize_frame(_image(), _frame(tuple(collapsed)), config=_config())
