import pytest

from face_symmetry_lab.application.quality import (
    QualityConfigError,
    QualityGate,
    QualityThresholds,
    evaluate_frame,
    validate_thresholds,
)
from face_symmetry_lab.domain.landmarks import build_landmark_frame
from face_symmetry_lab.domain.models import ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose

THRESHOLDS = QualityThresholds()


def _frame(
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    area: float = 0.25,
    face_count: int = 1,
    frame_id: int = 1,
):
    """ランドマークの外接矩形が画像の ``area`` を占めるフレーム。"""
    side = area**0.5
    low = (1.0 - side) / 2.0
    high = low + side
    points = ((low, low), (high, low), (low, high), (high, high)) if face_count == 1 else ()
    pose = FacePose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll) if face_count == 1 else None
    return build_landmark_frame(
        frame_id=frame_id,
        face_count=face_count,
        points=points,
        pose=pose,
        width=640,
        height=480,
    )


def test_a_frontal_well_sized_face_passes_the_gate() -> None:
    assert evaluate_frame(_frame(), THRESHOLDS, strict=True) == ()


def test_a_pc_webcam_arms_length_framing_is_not_rejected_as_too_far() -> None:
    """一般的なノート PC の webcam の距離で、フレーム幅の約 60% を占める顔は、

    外接矩形の面積比が 0.08 前後になる（拡縮した合成顔に対し、実際の Face
    Landmarker で確認済み）。ゲートを通すためだけに、利用者がそれより大きく
    顔を近づける必要があってはならない。
    """
    assert evaluate_frame(_frame(area=0.08), THRESHOLDS, strict=True) == ()


def test_a_seated_pc_webcam_framing_with_room_to_spare_is_not_rejected() -> None:
    """フレーム幅の約 30% を占める顔（上半身が入る、自然な着席姿勢）は、

    比が 0.02 前後になる。上の腕を伸ばした距離よりさらに遠いが、これも通らねば
    ならない。
    """
    assert evaluate_frame(_frame(area=0.02), THRESHOLDS, strict=True) == ()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"yaw": 20.0}, ReasonCode.OFF_AXIS_YAW),
        ({"pitch": -20.0}, ReasonCode.OFF_AXIS_PITCH),
        ({"roll": 30.0}, ReasonCode.OFF_AXIS_ROLL),
        ({"area": 0.005}, ReasonCode.TOO_FAR),
        ({"area": 0.80}, ReasonCode.TOO_CLOSE),
    ],
)
def test_each_violation_maps_to_its_reason_code(kwargs: dict, expected: ReasonCode) -> None:
    assert expected in evaluate_frame(_frame(**kwargs), THRESHOLDS, strict=True)


def test_several_violations_are_reported_together() -> None:
    reasons = evaluate_frame(_frame(yaw=25.0, pitch=25.0), THRESHOLDS, strict=True)

    assert ReasonCode.OFF_AXIS_YAW in reasons
    assert ReasonCode.OFF_AXIS_PITCH in reasons


def test_a_missing_pose_is_low_confidence() -> None:
    frame = build_landmark_frame(frame_id=1, face_count=1, points=((0.4, 0.4), (0.6, 0.6)))

    assert evaluate_frame(frame, THRESHOLDS, strict=True) == (ReasonCode.LOW_CONFIDENCE,)


def test_the_outer_threshold_is_more_permissive_than_the_inner_one() -> None:
    borderline = _frame(yaw=16.0)

    assert evaluate_frame(borderline, THRESHOLDS, strict=True) != ()
    assert evaluate_frame(borderline, THRESHOLDS, strict=False) == ()


def test_no_face_and_multi_face_short_circuit_the_gate() -> None:
    gate = QualityGate()

    empty = gate.update(_frame(face_count=0))
    crowd = gate.update(_frame(face_count=2))

    assert empty.status is Status.NO_FACE
    assert empty.reason_codes == (ReasonCode.NO_FACE,)
    assert crowd.status is Status.MULTI_FACE
    assert crowd.reason_codes == (ReasonCode.MULTIPLE_FACES,)


def test_five_consecutive_valid_frames_are_needed_to_reach_ok() -> None:
    gate = QualityGate()

    statuses = [gate.update(_frame(frame_id=index)).status for index in range(1, 6)]

    assert statuses == [Status.HOLD] * 4 + [Status.OK]


def test_frames_short_of_the_streak_report_unstable() -> None:
    gate = QualityGate()

    state = gate.update(_frame())

    assert state.status is Status.HOLD
    assert state.reason_codes == (ReasonCode.UNSTABLE,)


def test_the_documented_hold_ok_hold_sequence() -> None:
    """§21.3: HOLD -> 有効 5 回 -> OK -> outer 不成立 1 回 -> HOLD。"""
    gate = QualityGate()
    for index in range(1, 6):
        gate.update(_frame(frame_id=index))
    assert gate.status is Status.OK

    state = gate.update(_frame(yaw=20.0, frame_id=6))

    assert state.status is Status.HOLD
    assert ReasonCode.OFF_AXIS_YAW in state.reason_codes


def test_ok_survives_between_the_inner_and_outer_thresholds() -> None:
    gate = QualityGate()
    for index in range(1, 6):
        gate.update(_frame(frame_id=index))

    assert gate.update(_frame(yaw=16.0, frame_id=6)).status is Status.OK


def test_returning_to_ok_requires_the_full_streak_again() -> None:
    gate = QualityGate()
    for index in range(1, 6):
        gate.update(_frame(frame_id=index))
    gate.update(_frame(yaw=30.0, frame_id=6))

    statuses = [gate.update(_frame(frame_id=index)).status for index in range(7, 12)]

    assert statuses == [Status.HOLD] * 4 + [Status.OK]


def test_a_multi_face_frame_immediately_drops_ok() -> None:
    gate = QualityGate()
    for index in range(1, 6):
        gate.update(_frame(frame_id=index))

    assert gate.update(_frame(face_count=2, frame_id=6)).status is Status.MULTI_FACE


def test_reset_returns_the_gate_to_hold() -> None:
    gate = QualityGate()
    for index in range(1, 6):
        gate.update(_frame(frame_id=index))

    gate.reset()

    assert gate.status is Status.HOLD
    assert gate.update(_frame()).status is Status.HOLD


@pytest.mark.parametrize(
    "overrides",
    [
        {"yaw_inner_deg": 20.0, "yaw_outer_deg": 18.0},
        {"pitch_inner_deg": 0.0},
        {"roll_outer_deg": 60.0},
        {"face_area_min": 0.6, "face_area_max": 0.5},
        {"valid_frames_to_ok": 0},
        {"valid_frames_to_ok": 99},
    ],
)
def test_invalid_thresholds_are_refused(overrides: dict) -> None:
    with pytest.raises(QualityConfigError):
        validate_thresholds(QualityThresholds(**overrides))
