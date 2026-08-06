import numpy as np
import pytest

from face_symmetry_lab.domain.landmarks import LandmarkError, build_landmark_frame
from face_symmetry_lab.domain.overlay import (
    DEFAULT_MIDLINE_COLOR,
    DEFAULT_MIDLINE_THICKNESS,
    DEFAULT_POINT_COLOR,
    draw_face_midline,
    draw_landmark_points,
    draw_symmetry_deviation,
)


def _blank(height: int = 20, width: int = 20) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_landmarks_are_painted_at_the_expected_pixel() -> None:
    frame = build_landmark_frame(frame_id=1, face_count=1, points=((0.5, 0.5),))

    painted = draw_landmark_points(_blank(), frame, color=(0, 255, 0), radius=0)

    assert tuple(painted[10, 10]) == (0, 255, 0)
    assert painted.sum() == 255


def test_radius_paints_a_square_neighbourhood() -> None:
    frame = build_landmark_frame(frame_id=2, face_count=1, points=((0.5, 0.5),))

    painted = draw_landmark_points(_blank(), frame, color=(255, 255, 255), radius=1)

    assert np.all(painted[9:12, 9:12] == 255)
    assert np.all(painted[0:8, 0:8] == 0)


def test_source_image_is_never_mutated() -> None:
    source = _blank()
    frame = build_landmark_frame(frame_id=3, face_count=1, points=((0.5, 0.5),))

    painted = draw_landmark_points(source, frame, color=(255, 0, 0), radius=1)

    assert painted is not source
    assert source.sum() == 0


def test_frames_without_a_single_face_are_returned_unchanged() -> None:
    source = _blank()
    frame = build_landmark_frame(frame_id=4, face_count=0, points=())

    painted = draw_landmark_points(source, frame)

    assert painted.sum() == 0


def test_points_on_the_border_are_clamped_instead_of_wrapping() -> None:
    frame = build_landmark_frame(frame_id=5, face_count=1, points=((0.0, 0.0), (1.0, 1.0)))

    painted = draw_landmark_points(_blank(4, 4), frame, color=(255, 255, 255), radius=2)

    assert painted.shape == (4, 4, 3)
    assert np.all(painted[0, 0] == 255)
    assert np.all(painted[3, 3] == 255)
    # クランプされた描画が、反対側の隅までにじみ出してはならない。
    assert np.all(painted[0, 3] == 0)
    assert np.all(painted[3, 0] == 0)


def test_non_rgb_images_are_rejected() -> None:
    frame = build_landmark_frame(frame_id=6, face_count=1, points=((0.5, 0.5),))

    with pytest.raises(LandmarkError, match="RGB"):
        draw_landmark_points(np.zeros((8, 8), dtype=np.uint8), frame)


def test_tls_face_midline_is_drawn_through_the_face_extent() -> None:
    frame = build_landmark_frame(
        frame_id=7,
        face_count=1,
        points=((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5)),
        width=20,
        height=20,
    )

    painted = draw_face_midline(
        _blank(),
        frame,
        midline_indices=(0, 1, 2),
        color=(255, 0, 0),
        thickness=1,
    )

    assert np.all(painted[4:17, 10] == (255, 0, 0))


def test_midline_is_not_drawn_without_a_single_face() -> None:
    frame = build_landmark_frame(frame_id=8, face_count=0)

    painted = draw_face_midline(_blank(), frame, midline_indices=(0, 1, 2))

    assert painted.sum() == 0


def test_default_midline_is_thick_and_visually_distinct_from_landmarks() -> None:
    assert DEFAULT_MIDLINE_COLOR != DEFAULT_POINT_COLOR
    assert DEFAULT_MIDLINE_THICKNESS >= 5

    frame = build_landmark_frame(
        frame_id=9,
        face_count=1,
        points=((0.5, 0.2), (0.5, 0.5), (0.5, 0.8), (0.2, 0.5), (0.8, 0.5)),
        width=20,
        height=20,
    )
    painted = draw_face_midline(_blank(), frame, midline_indices=(0, 1, 2))

    line_pixels = np.all(painted == DEFAULT_MIDLINE_COLOR, axis=2)
    assert np.count_nonzero(line_pixels[10]) >= DEFAULT_MIDLINE_THICKNESS


def test_the_deviation_overlay_draws_both_layers_and_leaves_the_source_alone() -> None:
    """§8.6.2/FR-034: 実測点と対称化点、そしてその間を結ぶ線分。"""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    measured = np.array([[10.0, 10.0], [30.0, 10.0]], dtype=np.float64)
    symmetrized = np.array([[14.0, 10.0], [26.0, 10.0]], dtype=np.float64)

    painted = draw_symmetry_deviation(
        image,
        measured=measured,
        symmetrized=symmetrized,
        indices=(0, 1),
        measured_color=(255, 0, 0),
        symmetrized_color=(0, 255, 0),
        deviation_color=(0, 0, 255),
        radius=0,
    )

    assert np.array_equal(image, np.zeros((40, 40, 3), dtype=np.uint8))  # 純色
    assert tuple(painted[10, 10]) == (255, 0, 0)  # 実測
    assert tuple(painted[10, 14]) == (0, 255, 0)  # 対称化
    assert tuple(painted[10, 12]) == (0, 0, 255)  # ズレの線分


def test_the_deviation_overlay_skips_points_with_no_correspondence() -> None:
    """描かれるのは対称化した index だけである。残りは一度も折り返していない。"""
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    measured = np.array([[10.0, 10.0], [30.0, 30.0]], dtype=np.float64)
    symmetrized = measured.copy()

    painted = draw_symmetry_deviation(
        image,
        measured=measured,
        symmetrized=symmetrized,
        indices=(0,),
        radius=0,
    )

    assert not np.array_equal(painted[10, 10], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(painted[30, 30], np.zeros(3, dtype=np.uint8))


def test_the_deviation_overlay_clamps_points_outside_the_frame() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    measured = np.array([[-40.0, 5.0]], dtype=np.float64)
    symmetrized = np.array([[60.0, 5.0]], dtype=np.float64)

    painted = draw_symmetry_deviation(
        image, measured=measured, symmetrized=symmetrized, indices=(0,), radius=0
    )

    assert painted.shape == image.shape


def test_the_deviation_overlay_rejects_a_non_finite_coordinate() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    measured = np.array([[float("nan"), 5.0]], dtype=np.float64)
    symmetrized = np.array([[6.0, 5.0]], dtype=np.float64)

    with pytest.raises(LandmarkError):
        draw_symmetry_deviation(image, measured=measured, symmetrized=symmetrized, indices=(0,))


def test_the_deviation_overlay_rejects_an_out_of_range_index() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    points = np.array([[6.0, 5.0]], dtype=np.float64)

    with pytest.raises(LandmarkError):
        draw_symmetry_deviation(image, measured=points, symmetrized=points, indices=(3,))
