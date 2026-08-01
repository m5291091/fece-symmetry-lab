import numpy as np
import pytest

from face_symmetry_lab.application.analyzer import FrameAnalyzer
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.metrics import PART_NAMES, PartMap, build_metric_config
from face_symmetry_lab.domain.models import ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose

# Normalized landmarks of a symmetric face inside a 640x480 frame.
_BASE_POINTS = (
    (0.40, 0.40),  # 0 left eye outer
    (0.45, 0.40),  # 1 left eye inner
    (0.55, 0.40),  # 2 right eye inner
    (0.60, 0.40),  # 3 right eye outer
    (0.50, 0.30),  # 4 midline
    (0.50, 0.50),  # 5 midline
    (0.50, 0.62),  # 6 midline
    (0.35, 0.55),  # 7 contour
    (0.65, 0.55),  # 8
    (0.36, 0.45),  # 9
    (0.64, 0.45),  # 10
    (0.38, 0.35),  # 11
    (0.62, 0.35),  # 12
    (0.47, 0.50),  # 13 nose
    (0.53, 0.50),  # 14
    (0.46, 0.48),  # 15
    (0.54, 0.48),  # 16
    (0.48, 0.52),  # 17
    (0.52, 0.52),  # 18
    (0.44, 0.58),  # 19 mouth
    (0.56, 0.58),  # 20
    (0.47, 0.57),  # 21
    (0.53, 0.57),  # 22
    (0.48, 0.60),  # 23
    (0.52, 0.60),  # 24
    (0.42, 0.36),  # 25 brow
    (0.58, 0.36),  # 26
)


def _scaled(points, factor: float = 1.3):
    """Grow the face about the frame centre so it clears the face-area gate."""
    return tuple((0.5 + (x - 0.5) * factor, 0.5 + (y - 0.5) * factor) for x, y in points)


SYMMETRIC_POINTS = _scaled(_BASE_POINTS)


def _config():
    return build_metric_config(
        metric_version="symmetry-2d/test",
        landmark_map_version="test/1.0.0",
        landmark_count=len(SYMMETRIC_POINTS),
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8), (9, 10), (11, 12)), min_pairs=3),
            "eyes": PartMap(pairs=((0, 3), (1, 2), (25, 26)), min_pairs=3),
            "nose": PartMap(pairs=((13, 14), (15, 16), (17, 18)), min_pairs=3),
            "mouth": PartMap(pairs=((19, 20), (21, 22), (23, 24)), min_pairs=3),
        },
        sensitivity=dict.fromkeys(PART_NAMES, 30.0),
        weights=dict.fromkeys(PART_NAMES, 0.25),
        sensitivity_frozen=False,
    )


def _frame(points=SYMMETRIC_POINTS, *, frame_id=1, yaw=0.0, face_count=1):
    return build_landmark_frame(
        frame_id=frame_id,
        face_count=face_count,
        points=points if face_count == 1 else (),
        pose=FacePose(yaw_deg=yaw, pitch_deg=0.0, roll_deg=0.0) if face_count == 1 else None,
        width=640,
        height=480,
    )


def _analyzer(**kwargs) -> FrameAnalyzer:
    return FrameAnalyzer(
        config=_config(),
        skin_pairs=((7, 8), (9, 10), (11, 12)),
        **kwargs,
    )


def _drive_to_ok(
    analyzer: FrameAnalyzer,
    *,
    start: float = 0.0,
    rgb_image: np.ndarray | None = None,
):
    outcome = None
    for index in range(1, 6):
        outcome = analyzer.analyze(
            _frame(frame_id=index),
            at=start + index * 0.1,
            rgb_image=rgb_image,
        )
    assert outcome is not None
    return outcome


def test_no_score_is_produced_before_the_gate_opens() -> None:
    analyzer = _analyzer()

    outcome = analyzer.analyze(_frame(), at=0.0)

    assert outcome.state.status is Status.HOLD
    assert outcome.symmetry is None
    assert outcome.overall_raw is None
    assert outcome.overall_smooth is None


def test_a_symmetric_face_scores_once_the_gate_opens() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.state.status is Status.OK
    assert outcome.overall_raw == pytest.approx(100.0, abs=1e-6)
    assert outcome.symmetry is not None
    assert outcome.symmetry.part("mouth") == pytest.approx(100.0, abs=1e-6)


def test_the_smoothed_value_starts_at_the_first_raw_score() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.overall_smooth == pytest.approx(outcome.overall_raw, abs=1e-6)


def test_the_smoothed_value_lags_a_change_in_the_raw_score() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    skewed = list(SYMMETRIC_POINTS)
    skewed[19] = (0.20, SYMMETRIC_POINTS[19][1])
    outcome = analyzer.analyze(_frame(tuple(skewed), frame_id=6), at=0.65)

    assert outcome.overall_raw is not None
    assert outcome.overall_smooth is not None
    assert outcome.overall_raw < outcome.overall_smooth < 100.0


def test_an_off_axis_frame_stops_producing_scores() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    outcome = analyzer.analyze(_frame(frame_id=6, yaw=30.0), at=0.7)

    assert outcome.state.status is Status.HOLD
    assert ReasonCode.OFF_AXIS_YAW in outcome.state.reason_codes
    assert outcome.overall_raw is None


def test_multi_face_frames_never_score() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    outcome = analyzer.analyze(_frame(frame_id=6, face_count=2), at=0.7)

    assert outcome.state.status is Status.MULTI_FACE
    assert outcome.symmetry is None


def test_a_prolonged_face_loss_resets_the_smoothing() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    analyzer.analyze(_frame(frame_id=6, face_count=0), at=1.0)
    analyzer.analyze(_frame(frame_id=7, face_count=0), at=2.0)
    recovered = None
    for index in range(8, 13):
        recovered = analyzer.analyze(_frame(frame_id=index), at=2.0 + index * 0.1)

    assert recovered is not None
    assert recovered.overall_smooth == pytest.approx(recovered.overall_raw, abs=1e-6)


def test_degenerate_landmarks_become_an_error_rather_than_a_score() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    collapsed = list(SYMMETRIC_POINTS)
    collapsed[2] = collapsed[0]
    collapsed[3] = collapsed[1]
    outcome = analyzer.analyze(_frame(tuple(collapsed), frame_id=6), at=0.7)

    assert outcome.state.status is Status.ERROR
    assert outcome.symmetry is None
    assert outcome.overall_smooth is None


def test_reset_clears_the_gate_and_the_smoothing() -> None:
    analyzer = _analyzer()
    _drive_to_ok(analyzer)

    analyzer.reset()
    outcome = analyzer.analyze(_frame(frame_id=6), at=1.0)

    assert outcome.state.status is Status.HOLD
    assert outcome.overall_smooth is None


def test_the_outcome_carries_the_versions_needed_to_reproduce_it() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.metric_version == "symmetry-2d/test"
    assert outcome.landmark_map_version == "test/1.0.0"
    assert outcome.sensitivity_frozen is False


def test_the_outcome_reports_pose_and_face_area() -> None:
    outcome = _drive_to_ok(_analyzer())

    assert outcome.pose is not None
    assert outcome.pose.yaw_deg == pytest.approx(0.0)
    assert outcome.face_area_ratio is not None
    assert 0.0 < outcome.face_area_ratio < 1.0


def test_update_swaps_thresholds_and_resets_the_gate_and_smoothing() -> None:
    """§15.2: applying new Advanced settings must not keep stale EMA history."""
    analyzer = _analyzer()
    _drive_to_ok(analyzer)  # gate is OK and EMA has a value

    analyzer.update(thresholds=QualityThresholds(valid_frames_to_ok=1), tau_seconds=0.2)
    outcome = analyzer.analyze(_frame(frame_id=99), at=100.0)

    assert analyzer.thresholds.valid_frames_to_ok == 1
    assert outcome.state.status is Status.OK  # new threshold applies immediately
    assert outcome.overall_smooth == outcome.overall_raw  # smoothing history was reset


def test_custom_thresholds_change_how_quickly_the_gate_opens() -> None:
    analyzer = _analyzer(thresholds=QualityThresholds(valid_frames_to_ok=1))

    outcome = analyzer.analyze(_frame(), at=0.0)

    assert outcome.state.status is Status.OK


def test_skin_surface_reference_is_separate_from_the_geometry_overall() -> None:
    mirrored = np.full((480, 640, 3), (150, 120, 105), dtype=np.uint8)
    analyzer = _analyzer()

    baseline = _drive_to_ok(analyzer, rgb_image=mirrored)
    changed_image = mirrored.copy()
    rng = np.random.default_rng(42)
    changed_image[250:290, 180:220] = rng.integers(
        0,
        256,
        size=(40, 40, 3),
        dtype=np.uint8,
    )
    changed = analyzer.analyze(_frame(frame_id=6), at=0.7, rgb_image=changed_image)

    assert baseline.skin_quality is not None
    assert baseline.skin_quality.score == pytest.approx(100.0, abs=1e-6)
    assert changed.skin_quality is not None
    assert changed.skin_quality.score is not None
    assert changed.skin_quality.score < baseline.skin_quality.score
    assert changed.overall_raw == pytest.approx(baseline.overall_raw, abs=1e-6)
