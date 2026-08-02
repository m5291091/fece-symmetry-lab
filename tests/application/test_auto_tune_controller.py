import numpy as np
import pytest

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.auto_tune_controller import (
    CADENCE_RETUNE_INTERVAL_SECONDS,
    DISPLAY_RETUNE_INTERVAL_SECONDS,
    MIN_CADENCE_SAMPLES,
    MIN_SENSITIVITY_SAMPLES,
    SENSITIVITY_RETUNE_INTERVAL_SECONDS,
    AutoTuneController,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PART_NAMES, PartMap, build_metric_config
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.domain.symmetry import SymmetryResult

#: Spread out so `face_area_ratio` is a usable non-degenerate fraction.
SPREAD_POINTS = ((0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6))


def _controller(*, enabled: bool = True) -> AutoTuneController:
    return AutoTuneController(enabled=enabled, started_at=0.0)


def _frame(points: tuple[tuple[float, float], ...] = SPREAD_POINTS):
    return build_landmark_frame(
        frame_id=1,
        face_count=1 if points else 0,
        points=points,
        width=100,
        height=100,
    )


def _flat_rgb(value: int) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def _metric_config(sensitivity: float = 30.0):
    return build_metric_config(
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
        sensitivity=dict.fromkeys(PART_NAMES, sensitivity),
        weights=dict.fromkeys(PART_NAMES, 0.25),
        sensitivity_frozen=True,
    )


def _outcome(distances: dict[str, float]) -> AnalysisOutcome:
    symmetry = SymmetryResult(
        scores=dict.fromkeys(PART_NAMES, 90.0),
        distances=dict(distances),
        overall=90.0,
        metric_version="symmetry/test",
        landmark_map_version="map/test",
    )
    return AnalysisOutcome(
        frame_id=1,
        state=AnalysisState(status=Status.OK),
        symmetry=symmetry,
        overall_smooth=90.0,
        pose=None,
        face_area_ratio=0.04,
        metric_version="symmetry/test",
        landmark_map_version="map/test",
        sensitivity_frozen=True,
    )


def test_a_disabled_controller_recommends_nothing_at_all() -> None:
    """AC-AUTOTUNE-01/02: with auto-tuning off nothing is ever recommended."""
    controller = _controller(enabled=False)

    display = controller.display_tuning(_flat_rgb(64), _frame(), at=0.0)

    assert display.brightness_gain is None
    assert display.zoom_policy is None
    assert controller.enabled is False


def test_display_tuning_recommends_a_gain_from_the_raw_frame_luminance() -> None:
    tuning = _controller().display_tuning(_flat_rgb(64), _frame(), at=0.0)

    assert tuning.brightness_gain == pytest.approx(2.0)  # 128 target / 64 measured
    assert tuning.zoom_policy is not None


def test_brightness_and_zoom_share_one_throttle() -> None:
    """Both come from the same frame at the same instant, so a second call
    inside the interval must not re-recommend just one of them."""
    controller = _controller()

    controller.display_tuning(_flat_rgb(64), _frame(), at=0.0)
    within = controller.display_tuning(
        _flat_rgb(64), _frame(), at=DISPLAY_RETUNE_INTERVAL_SECONDS / 2
    )
    after = controller.display_tuning(_flat_rgb(64), _frame(), at=DISPLAY_RETUNE_INTERVAL_SECONDS)

    assert within.brightness_gain is None
    assert within.zoom_policy is None
    assert after.brightness_gain is not None
    assert after.zoom_policy is not None


def test_an_unusable_frame_leaves_the_gain_alone_without_raising() -> None:
    """AC-AUTOTUNE-01: a failed measurement is swallowed as "no change"."""
    tuning = _controller().display_tuning(_flat_rgb(0), _frame(), at=0.0)

    assert tuning.brightness_gain is None  # a pure-black frame has no valid gain


def test_zoom_is_not_recommended_without_a_confirmed_single_face() -> None:
    tuning = _controller().display_tuning(_flat_rgb(64), _frame(points=()), at=0.0)

    assert tuning.zoom_policy is None
    assert tuning.brightness_gain is not None  # brightness still works


def test_cadence_needs_a_minimum_number_of_samples_before_recommending() -> None:
    controller = _controller()

    for index in range(MIN_CADENCE_SAMPLES - 1):
        recommended = controller.cadence_tuning(
            300.0, at=index * CADENCE_RETUNE_INTERVAL_SECONDS, current_interval_ms=125
        )
        assert recommended is None


def test_a_sustained_heavy_load_recommends_a_slower_interval() -> None:
    controller = _controller()
    at = 0.0
    recommended = None

    for _ in range(MIN_CADENCE_SAMPLES + 1):
        recommended = controller.cadence_tuning(300.0, at=at, current_interval_ms=125)
        at += CADENCE_RETUNE_INTERVAL_SECONDS

    assert recommended == 333


def test_cadence_returns_none_when_the_recommendation_already_holds() -> None:
    """AC-AUTOTUNE-02: an unchanged answer must not churn the recorder."""
    controller = _controller()
    at = 0.0
    recommended = None

    for _ in range(MIN_CADENCE_SAMPLES + 1):
        recommended = controller.cadence_tuning(300.0, at=at, current_interval_ms=333)
        at += CADENCE_RETUNE_INTERVAL_SECONDS

    assert recommended is None


def test_sensitivity_needs_enough_samples_and_a_real_change() -> None:
    controller = _controller()
    config = _metric_config()
    steady = dict.fromkeys(PART_NAMES, 0.1)
    at = 0.0
    recommended = None

    for _ in range(MIN_SENSITIVITY_SAMPLES + 2):
        recommended = controller.sensitivity_tuning(_outcome(steady), at=at, config=config)
        at += SENSITIVITY_RETUNE_INTERVAL_SECONDS

    # A perfectly steady distance has no jitter to damp, so nothing moves.
    assert recommended is None


def test_a_jittery_part_lowers_its_sensitivity_and_leaves_the_others_alone() -> None:
    controller = _controller()
    config = _metric_config(sensitivity=30.0)
    at = 0.0
    recommended = None

    for index in range(MIN_SENSITIVITY_SAMPLES + 2):
        distances = dict.fromkeys(PART_NAMES, 0.1)
        distances["contour"] = 0.05 if index % 2 == 0 else 0.35
        recommended = controller.sensitivity_tuning(_outcome(distances), at=at, config=config)
        at += SENSITIVITY_RETUNE_INTERVAL_SECONDS

    assert recommended is not None
    assert recommended["contour"] < 30.0
    assert recommended["eyes"] == pytest.approx(30.0)


def test_a_disabled_controller_never_recommends_cadence_or_sensitivity() -> None:
    controller = _controller(enabled=False)
    config = _metric_config()
    at = 0.0

    for _ in range(MIN_CADENCE_SAMPLES + MIN_SENSITIVITY_SAMPLES + 2):
        assert controller.cadence_tuning(300.0, at=at, current_interval_ms=125) is None
        assert (
            controller.sensitivity_tuning(
                _outcome(dict.fromkeys(PART_NAMES, 0.1)), at=at, config=config
            )
            is None
        )
        at += SENSITIVITY_RETUNE_INTERVAL_SECONDS
