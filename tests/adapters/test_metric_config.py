import json
from pathlib import Path

import pytest

from face_symmetry_lab.adapters.io.metric_config import (
    DEFAULT_METRIC_CONFIG_PATH,
    load_metric_config,
)
from face_symmetry_lab.domain.landmarks import FACE_LANDMARK_COUNT
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfigError


def test_the_frozen_repository_bundle_loads() -> None:
    config = load_metric_config()

    assert config.metric_version.startswith("symmetry-2d/")
    assert config.landmark_count == FACE_LANDMARK_COUNT
    assert set(config.parts) == set(PART_NAMES)


def test_the_frozen_bundle_carries_no_calibratable_coefficient() -> None:
    """AC-SCORE-04: §8.6 left the metric with no free parameter to freeze."""
    raw = json.loads(DEFAULT_METRIC_CONFIG_PATH.read_text(encoding="utf-8"))

    assert "sensitivity" not in raw
    assert "weights" not in raw
    assert "sensitivity_frozen" not in raw


def test_the_frozen_bundle_pools_every_pair_for_the_overall_distance() -> None:
    """§8.6: the overall fold sees each of the 63 correspondences once."""
    config = load_metric_config()

    pooled = config.all_pairs

    assert len(pooled) == sum(len(config.parts[name].pairs) for name in PART_NAMES)
    assert len(pooled) == len(set(pooled))
    assert config.overall_min_pairs == sum(config.parts[name].min_pairs for name in PART_NAMES)
    assert config.overall_min_pairs < len(pooled)


def test_the_frozen_map_has_enough_pairs_for_every_part() -> None:
    config = load_metric_config()

    for name in PART_NAMES:
        part = config.parts[name]
        assert len(part.pairs) >= part.min_pairs >= 3


def test_the_frozen_map_never_reuses_a_landmark() -> None:
    config = load_metric_config()

    seen: list[int] = []
    for name in PART_NAMES:
        for left, right in config.parts[name].pairs:
            seen.extend((left, right))

    assert len(seen) == len(set(seen))


def test_the_frozen_map_keeps_midline_points_out_of_the_pairs() -> None:
    config = load_metric_config()

    paired = {index for name in PART_NAMES for pair in config.parts[name].pairs for index in pair}

    assert paired.isdisjoint(set(config.midline_indices))


def test_the_frozen_eye_anchors_are_not_iris_points() -> None:
    """§8.3 forbids the gaze-dependent iris centres as a scale reference."""
    iris_indices = set(range(468, 478))
    config = load_metric_config()

    assert set(config.left_eye_corners).isdisjoint(iris_indices)
    assert set(config.right_eye_corners).isdisjoint(iris_indices)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(MetricConfigError, match="metric config"):
        load_metric_config(tmp_path / "absent.json")


def test_a_malformed_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "metric.json"
    path.write_text("{oops", encoding="utf-8")

    with pytest.raises(MetricConfigError, match="metric config"):
        load_metric_config(path)


def test_a_bundle_with_a_reused_landmark_is_refused(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_METRIC_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["parts"]["nose"]["pairs"][0] = raw["parts"]["mouth"]["pairs"][0]
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(MetricConfigError, match="重複"):
        load_metric_config(path)


def test_a_bundle_missing_a_part_is_refused(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_METRIC_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["parts"]["mouth"]
    path = tmp_path / "metric.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(MetricConfigError, match="部位"):
        load_metric_config(path)
