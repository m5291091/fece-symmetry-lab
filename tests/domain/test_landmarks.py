import math

import numpy as np
import pytest

from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    build_landmark_frame,
    to_pixel_array,
    to_pixel_array_3d,
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


# --------------------------------------------------------------------------
# 推定奥行き z（§8.1.1）
# --------------------------------------------------------------------------


def _frame_2d(**kwargs) -> LandmarkFrame:
    return build_landmark_frame(
        frame_id=10,
        face_count=1,
        points=((0.25, 0.5), (0.75, 0.25)),
        width=100,
        height=200,
        **kwargs,
    )


def test_depths_ride_along_with_the_points() -> None:
    frame = _frame_2d(depths=(-0.02, 0.03))

    assert frame.depths == (-0.02, 0.03)
    assert frame.has_depth is True


def test_a_frame_without_depths_reports_the_depth_as_missing() -> None:
    frame = _frame_2d()

    assert frame.depths is None
    assert frame.has_depth is False


def test_depths_must_be_supplied_for_every_landmark() -> None:
    with pytest.raises(LandmarkError, match="奥行き"):
        _frame_2d(depths=(-0.02,))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_depths_are_rejected(bad: float) -> None:
    with pytest.raises(LandmarkError, match="奥行き"):
        _frame_2d(depths=(0.0, bad))


def test_a_frame_without_a_single_face_cannot_carry_depths() -> None:
    with pytest.raises(LandmarkError, match="奥行き"):
        build_landmark_frame(frame_id=11, face_count=2, points=(), depths=(0.0,))


def test_the_pixel_array_stays_two_dimensional_when_depths_are_present() -> None:
    """既存の呼び出し側はすべて 2D 前提である。z を足しても形を変えない。"""
    without_depth = to_pixel_array(_frame_2d())
    with_depth = to_pixel_array(_frame_2d(depths=(-0.02, 0.03)))

    assert without_depth.shape == (2, 2)
    assert with_depth.shape == (2, 2)
    assert np.array_equal(with_depth, without_depth)


def test_the_3d_pixel_array_scales_depth_with_the_width_like_x() -> None:
    """§8.1.1: z は x と同じ係数（画像の幅）で画素相当へ戻す。"""
    frame = _frame_2d(depths=(-0.02, 0.03))

    pixels = to_pixel_array_3d(frame)

    assert pixels.shape == (2, 3)
    assert pixels[0] == pytest.approx((25.0, 100.0, -2.0))
    assert pixels[1] == pytest.approx((75.0, 50.0, 3.0))


def test_the_3d_pixel_array_agrees_with_the_2d_one_on_x_and_y() -> None:
    frame = _frame_2d(depths=(-0.02, 0.03))

    assert np.array_equal(to_pixel_array_3d(frame)[:, :2], to_pixel_array(frame))


def test_the_3d_pixel_array_refuses_a_frame_whose_depth_is_missing() -> None:
    with pytest.raises(LandmarkError, match="奥行き"):
        to_pixel_array_3d(_frame_2d())


def test_the_3d_pixel_array_needs_the_image_size() -> None:
    frame = build_landmark_frame(
        frame_id=12,
        face_count=1,
        points=((0.5, 0.5),),
        depths=(0.01,),
    )

    with pytest.raises(LandmarkError, match="画像サイズ"):
        to_pixel_array_3d(frame)
