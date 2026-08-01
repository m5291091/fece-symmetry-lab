import math

import pytest

from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    build_landmark_frame,
    to_pixel_points,
)


def test_single_face_frame_keeps_normalized_points() -> None:
    frame = build_landmark_frame(
        frame_id=1,
        face_count=1,
        points=((0.25, 0.5), (0.75, 0.5)),
    )

    assert isinstance(frame, LandmarkFrame)
    assert frame.has_single_face is True
    assert frame.points == ((0.25, 0.5), (0.75, 0.5))


@pytest.mark.parametrize("face_count", [0, 2, 3])
def test_frames_without_exactly_one_face_carry_no_points(face_count: int) -> None:
    frame = build_landmark_frame(frame_id=2, face_count=face_count, points=())

    assert frame.has_single_face is False
    assert frame.points == ()


def test_multi_face_frame_refuses_to_carry_one_faces_points() -> None:
    with pytest.raises(LandmarkError, match="単一顔"):
        build_landmark_frame(frame_id=3, face_count=2, points=((0.5, 0.5),))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_are_rejected(bad: float) -> None:
    with pytest.raises(LandmarkError, match="有限"):
        build_landmark_frame(frame_id=4, face_count=1, points=((0.5, bad),))


def test_negative_frame_id_is_rejected() -> None:
    with pytest.raises(LandmarkError, match="frame_id"):
        build_landmark_frame(frame_id=-1, face_count=1, points=((0.5, 0.5),))


def test_negative_face_count_is_rejected() -> None:
    with pytest.raises(LandmarkError, match="face_count"):
        build_landmark_frame(frame_id=0, face_count=-1, points=())


def test_pixel_points_scale_with_width_and_height() -> None:
    frame = build_landmark_frame(
        frame_id=5,
        face_count=1,
        points=((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
    )

    pixels = to_pixel_points(frame, width=100, height=200)

    assert pixels == ((0, 0), (50, 100), (99, 199))


def test_pixel_points_clamp_slightly_out_of_range_landmarks() -> None:
    frame = build_landmark_frame(
        frame_id=6,
        face_count=1,
        points=((-0.2, 1.4),),
    )

    assert to_pixel_points(frame, width=10, height=10) == ((0, 9),)


def test_pixel_points_are_empty_without_a_single_face() -> None:
    frame = build_landmark_frame(frame_id=7, face_count=2, points=())

    assert to_pixel_points(frame, width=10, height=10) == ()


@pytest.mark.parametrize(("width", "height"), [(0, 10), (10, 0), (-1, 10)])
def test_pixel_points_reject_non_positive_frame_size(width: int, height: int) -> None:
    frame = build_landmark_frame(frame_id=8, face_count=1, points=((0.5, 0.5),))

    with pytest.raises(LandmarkError, match="画像サイズ"):
        to_pixel_points(frame, width=width, height=height)
