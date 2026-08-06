import pytest

from face_symmetry_lab.domain.metrics import (
    PART_NAMES,
    MetricConfigError,
    PartMap,
    build_metric_config,
)


def _config(**overrides):
    base = {
        "metric_version": "symmetry-2d/2.0.0",
        "landmark_map_version": "mediapipe-478-mirror/1.0.0",
        "landmark_count": 478,
        "left_eye_corners": (33, 133),
        "right_eye_corners": (263, 362),
        "midline_indices": (10, 151, 4, 0, 17, 152),
        "parts": {
            "contour": PartMap(pairs=((21, 251), (54, 284), (103, 332)), min_pairs=3),
            "eyes": PartMap(pairs=((33, 263), (133, 362), (159, 386)), min_pairs=3),
            "nose": PartMap(pairs=((115, 344), (49, 279), (48, 278)), min_pairs=2),
            "mouth": PartMap(pairs=((61, 291), (78, 308), (39, 269)), min_pairs=2),
        },
    }
    base.update(overrides)
    return base


def test_a_valid_configuration_is_accepted() -> None:
    config = build_metric_config(**_config())

    assert config.metric_version == "symmetry-2d/2.0.0"
    assert set(config.parts) == set(PART_NAMES)


def test_the_bundle_carries_no_sensitivity_or_weight_parameters() -> None:
    """§8.6: 距離指標には、較正すべき自由な係数がもう残っていない。"""
    config = build_metric_config(**_config())

    assert not hasattr(config, "sensitivity")
    assert not hasattr(config, "weights")
    assert not hasattr(config, "sensitivity_frozen")


def test_all_pairs_pools_every_part_exactly_once() -> None:
    """§8.6 の総合距離はまとめた集合を 1 回折りたたむので、重複があってはならない。"""
    config = build_metric_config(**_config())

    pooled = config.all_pairs

    expected = tuple(pair for name in PART_NAMES for pair in config.parts[name].pairs)
    assert pooled == expected
    assert len(pooled) == len(set(pooled))


def test_the_overall_minimum_is_the_sum_of_the_part_minimums() -> None:
    config = build_metric_config(**_config())

    assert config.overall_min_pairs == 3 + 3 + 2 + 2


def test_every_part_must_be_present() -> None:
    parts = {"contour": PartMap(pairs=((21, 251),), min_pairs=1)}

    with pytest.raises(MetricConfigError, match="部位"):
        build_metric_config(**_config(parts=parts))


def test_self_pairs_are_rejected() -> None:
    parts = dict(_config()["parts"])
    parts["nose"] = PartMap(pairs=((115, 115),), min_pairs=1)

    with pytest.raises(MetricConfigError, match="自己対応"):
        build_metric_config(**_config(parts=parts))


def test_duplicate_pairs_are_rejected() -> None:
    parts = dict(_config()["parts"])
    parts["mouth"] = PartMap(pairs=((61, 291), (61, 291)), min_pairs=1)

    with pytest.raises(MetricConfigError, match="重複"):
        build_metric_config(**_config(parts=parts))


def test_indices_outside_the_landmark_range_are_rejected() -> None:
    parts = dict(_config()["parts"])
    parts["eyes"] = PartMap(pairs=((33, 999),), min_pairs=1)

    with pytest.raises(MetricConfigError, match="範囲"):
        build_metric_config(**_config(parts=parts))


def test_a_landmark_may_not_be_reused_across_sides() -> None:
    parts = dict(_config()["parts"])
    parts["eyes"] = PartMap(pairs=((33, 263), (263, 133)), min_pairs=1)

    with pytest.raises(MetricConfigError, match="重複"):
        build_metric_config(**_config(parts=parts))


def test_min_pairs_may_not_exceed_the_available_pairs() -> None:
    parts = dict(_config()["parts"])
    parts["nose"] = PartMap(pairs=((115, 344),), min_pairs=5)

    with pytest.raises(MetricConfigError, match="min_pairs"):
        build_metric_config(**_config(parts=parts))


def test_the_midline_needs_at_least_three_points_for_a_line_fit() -> None:
    with pytest.raises(MetricConfigError, match="正中"):
        build_metric_config(**_config(midline_indices=(10, 152)))


def test_eye_corner_indices_must_differ() -> None:
    with pytest.raises(MetricConfigError, match="眼"):
        build_metric_config(**_config(left_eye_corners=(33, 33)))
