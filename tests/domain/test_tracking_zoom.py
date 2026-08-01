import pytest

from face_symmetry_lab.domain.tracking_zoom import (
    FULL_FRAME,
    BoundingBox,
    TrackingZoomError,
    TrackingZoomPolicy,
    bounding_box_of,
    next_zoom_window,
    remap_point,
    remap_points,
    resize_scale_for,
)


def test_bounding_box_rejects_a_degenerate_or_out_of_range_rectangle() -> None:
    with pytest.raises(TrackingZoomError):
        BoundingBox(0.5, 0.0, 0.5, 1.0)
    with pytest.raises(TrackingZoomError):
        BoundingBox(-0.1, 0.0, 0.5, 1.0)
    with pytest.raises(TrackingZoomError):
        BoundingBox(0.0, 0.0, 1.1, 1.0)


def test_bounding_box_width_and_height() -> None:
    box = BoundingBox(0.2, 0.3, 0.6, 0.5)
    assert box.width == pytest.approx(0.4)
    assert box.height == pytest.approx(0.2)


def test_no_prior_face_means_search_the_full_frame() -> None:
    assert next_zoom_window(None) == FULL_FRAME


def test_a_small_bbox_is_expanded_by_the_margin_and_floored_at_min_window() -> None:
    # width=height=0.05 is small enough that the 0.6 margin still lands
    # below the 0.18 floor, so the floor should win.
    bbox = BoundingBox(0.475, 0.475, 0.525, 0.525)
    window = next_zoom_window(bbox, policy=TrackingZoomPolicy())

    assert window.width == pytest.approx(0.18)
    assert window.height == pytest.approx(0.18)
    # Centered on the bbox's own center.
    assert window.x_min == pytest.approx(0.5 - 0.09)
    assert window.y_min == pytest.approx(0.5 - 0.09)


def test_a_large_bbox_uses_the_margin_expansion_not_the_floor() -> None:
    bbox = BoundingBox(0.3, 0.3, 0.5, 0.5)  # width=height=0.2
    window = next_zoom_window(bbox, policy=TrackingZoomPolicy(margin_ratio=0.5))

    assert window.width == pytest.approx(0.3)  # 0.2 * 1.5
    assert window.height == pytest.approx(0.3)


def test_the_window_clamps_to_the_frame_edge_instead_of_going_out_of_bounds() -> None:
    bbox = BoundingBox(0.0, 0.0, 0.1, 0.1)  # hugging the top-left corner
    window = next_zoom_window(bbox, policy=TrackingZoomPolicy(margin_ratio=0.6))

    assert window.x_min == pytest.approx(0.0)
    assert window.y_min == pytest.approx(0.0)
    assert window.x_max <= 1.0
    assert window.y_max <= 1.0


def test_bounding_box_of_empty_points_is_none() -> None:
    assert bounding_box_of(()) is None


def test_bounding_box_of_a_degenerate_single_point_is_none() -> None:
    assert bounding_box_of(((0.4, 0.4), (0.4, 0.4))) is None


def test_bounding_box_of_computes_the_tight_rectangle() -> None:
    points = ((0.2, 0.3), (0.6, 0.3), (0.2, 0.7), (0.6, 0.7))
    box = bounding_box_of(points)

    assert box == BoundingBox(0.2, 0.3, 0.6, 0.7)


def test_remap_point_projects_crop_local_fractions_into_frame_fractions() -> None:
    window = BoundingBox(0.25, 0.25, 0.75, 0.75)  # centered quarter-frame crop

    assert remap_point((0.0, 0.0), window) == pytest.approx((0.25, 0.25))
    assert remap_point((1.0, 1.0), window) == pytest.approx((0.75, 0.75))
    assert remap_point((0.5, 0.5), window) == pytest.approx((0.5, 0.5))


def test_remap_points_preserves_order_and_count() -> None:
    window = BoundingBox(0.0, 0.0, 0.5, 0.5)
    points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    remapped = remap_points(points, window)

    assert remapped == ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5))


def test_resize_scale_for_upscales_a_small_crop_up_to_the_target() -> None:
    scale = resize_scale_for(120, 120, policy=TrackingZoomPolicy(target_px=480, max_scale=4.0))

    assert scale == pytest.approx(4.0)  # 480/120 == 4.0, within the cap


def test_resize_scale_for_never_downscales_a_crop_already_larger_than_target() -> None:
    scale = resize_scale_for(1280, 720, policy=TrackingZoomPolicy(target_px=480))

    assert scale == pytest.approx(1.0)


def test_resize_scale_for_caps_at_max_scale_instead_of_amplifying_noise() -> None:
    scale = resize_scale_for(10, 10, policy=TrackingZoomPolicy(target_px=480, max_scale=4.0))

    assert scale == pytest.approx(4.0)


def test_resize_scale_for_rejects_non_positive_pixel_sizes() -> None:
    with pytest.raises(TrackingZoomError):
        resize_scale_for(0, 100)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"margin_ratio": -0.1},
        {"margin_ratio": 3.1},
        {"min_window_size": 0.0},
        {"min_window_size": 1.1},
        {"recheck_interval_ms": -1},
        {"target_px": 0},
        {"max_scale": 0.9},
    ],
)
def test_invalid_policy_values_are_refused(kwargs: dict) -> None:
    with pytest.raises(TrackingZoomError):
        TrackingZoomPolicy(**kwargs)
