import pytest

from face_symmetry_lab.domain.metrics import (
    PART_NAMES,
    MetricConfigError,
    PartMap,
    build_metric_config,
    with_tunables,
)


def _config(**overrides):
    base = {
        "metric_version": "symmetry-2d/1.0.0",
        "landmark_map_version": "mediapipe-478-mirror/1.0.0",
        "landmark_count": 478,
        "left_eye_corners": (33, 133),
        "right_eye_corners": (263, 362),
        "midline_indices": (10, 151, 4, 0, 17, 152),
        "parts": {
            "contour": PartMap(pairs=((21, 251), (54, 284), (103, 332)), min_pairs=3),
            "eyes": PartMap(pairs=((33, 263), (133, 362), (159, 386)), min_pairs=3),
            "nose": PartMap(pairs=((115, 344), (49, 279), (48, 278)), min_pairs=3),
            "mouth": PartMap(pairs=((61, 291), (78, 308), (39, 269)), min_pairs=3),
        },
        "sensitivity": {name: 30.0 for name in PART_NAMES},
        "weights": {name: 0.25 for name in PART_NAMES},
        "sensitivity_frozen": False,
    }
    base.update(overrides)
    return base


def test_a_valid_configuration_is_accepted() -> None:
    config = build_metric_config(**_config())

    assert config.metric_version == "symmetry-2d/1.0.0"
    assert set(config.parts) == set(PART_NAMES)
    assert config.sensitivity_frozen is False


def test_weights_must_sum_to_one() -> None:
    weights = {"contour": 0.3, "eyes": 0.3, "nose": 0.3, "mouth": 0.3}

    with pytest.raises(MetricConfigError, match="重み"):
        build_metric_config(**_config(weights=weights))


def test_negative_weights_are_rejected() -> None:
    weights = {"contour": -0.1, "eyes": 0.4, "nose": 0.35, "mouth": 0.35}

    with pytest.raises(MetricConfigError, match="重み"):
        build_metric_config(**_config(weights=weights))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_sensitivity_must_be_positive(bad: float) -> None:
    sensitivity = {name: 30.0 for name in PART_NAMES}
    sensitivity["nose"] = bad

    with pytest.raises(MetricConfigError, match="感度"):
        build_metric_config(**_config(sensitivity=sensitivity))


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


def test_with_tunables_overrides_weights_and_sensitivity_only() -> None:
    """§15.2: editing weights/sensitivity must leave every other field pinned."""
    config = build_metric_config(**_config())
    new_weights = {"contour": 0.1, "eyes": 0.5, "nose": 0.3, "mouth": 0.1}
    new_sensitivity = {"contour": 10.0, "eyes": 20.0, "nose": 30.0, "mouth": 40.0}

    tuned = with_tunables(config, weights=new_weights, sensitivity=new_sensitivity)

    assert tuned.weights == new_weights
    assert tuned.sensitivity == new_sensitivity
    assert tuned.metric_version == config.metric_version
    assert tuned.landmark_map_version == config.landmark_map_version
    assert tuned.parts == config.parts
    assert tuned.midline_indices == config.midline_indices
    assert tuned.left_eye_corners == config.left_eye_corners
    assert tuned.right_eye_corners == config.right_eye_corners


def test_with_tunables_marks_the_result_as_no_longer_frozen() -> None:
    """An edited bundle can never claim the frozen calibration's provenance."""
    config = build_metric_config(**_config(sensitivity_frozen=True))

    tuned = with_tunables(
        config,
        weights={name: 0.25 for name in PART_NAMES},
        sensitivity={name: 30.0 for name in PART_NAMES},
    )

    assert tuned.sensitivity_frozen is False


def test_with_tunables_still_rejects_an_invalid_weight_sum() -> None:
    config = build_metric_config(**_config())

    with pytest.raises(MetricConfigError, match="重み"):
        with_tunables(
            config,
            weights={"contour": 0.3, "eyes": 0.3, "nose": 0.3, "mouth": 0.3},
            sensitivity={name: 30.0 for name in PART_NAMES},
        )


def test_with_tunables_still_rejects_non_positive_sensitivity() -> None:
    config = build_metric_config(**_config())

    with pytest.raises(MetricConfigError, match="感度"):
        with_tunables(
            config,
            weights={name: 0.25 for name in PART_NAMES},
            sensitivity={"contour": 0.0, "eyes": 30.0, "nose": 30.0, "mouth": 30.0},
        )
