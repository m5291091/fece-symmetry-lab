"""Quality gate and OK/HOLD hysteresis (§6.2, §6.4, §8.2).

The gate decides whether a frame may produce a score. It never scores, and it
keeps blocking reasons (`reason_codes`) apart from non-blocking notes
(`warning_codes`).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status

MAX_ANGLE_DEG: float = 45.0


class QualityConfigError(ValueError):
    """Invalid gate thresholds."""


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """§15.2 Advanced settings. Inner admits, outer releases."""

    yaw_inner_deg: float = 15.0
    yaw_outer_deg: float = 17.0
    pitch_inner_deg: float = 15.0
    pitch_outer_deg: float = 17.0
    roll_inner_deg: float = 25.0
    roll_outer_deg: float = 28.0
    face_area_min: float = 0.12
    face_area_max: float = 0.55
    valid_frames_to_ok: int = 5


def validate_thresholds(thresholds: QualityThresholds) -> QualityThresholds:
    pairs = (
        ("yaw", thresholds.yaw_inner_deg, thresholds.yaw_outer_deg),
        ("pitch", thresholds.pitch_inner_deg, thresholds.pitch_outer_deg),
        ("roll", thresholds.roll_inner_deg, thresholds.roll_outer_deg),
    )
    for name, inner, outer in pairs:
        if not (isfinite(inner) and isfinite(outer)):
            raise QualityConfigError(f"{name} の閾値は有限値である必要があります。")
        if not 0 < inner < outer <= MAX_ANGLE_DEG:
            raise QualityConfigError(f"{name} は 0 < inner < outer <= 45 が必要です。")

    if not 0 < thresholds.face_area_min < thresholds.face_area_max < 1:
        raise QualityConfigError("顔面積比は 0 < min < max < 1 が必要です。")
    if not 1 <= thresholds.valid_frames_to_ok <= 30:
        raise QualityConfigError("valid_frames_to_ok は 1〜30 が必要です。")
    return thresholds


def evaluate_frame(
    frame: LandmarkFrame,
    thresholds: QualityThresholds,
    *,
    strict: bool,
) -> tuple[ReasonCode, ...]:
    """Return the blocking reasons for one frame. Empty means the frame passes."""
    pose = frame.pose
    if pose is None:
        return (ReasonCode.LOW_CONFIDENCE,)

    yaw_limit = thresholds.yaw_inner_deg if strict else thresholds.yaw_outer_deg
    pitch_limit = thresholds.pitch_inner_deg if strict else thresholds.pitch_outer_deg
    roll_limit = thresholds.roll_inner_deg if strict else thresholds.roll_outer_deg

    reasons: list[ReasonCode] = []
    if abs(pose.yaw_deg) > yaw_limit:
        reasons.append(ReasonCode.OFF_AXIS_YAW)
    if abs(pose.pitch_deg) > pitch_limit:
        reasons.append(ReasonCode.OFF_AXIS_PITCH)
    if abs(pose.roll_deg) > roll_limit:
        reasons.append(ReasonCode.OFF_AXIS_ROLL)

    area = frame.face_area_ratio
    if area is None:
        reasons.append(ReasonCode.LOW_CONFIDENCE)
    elif area < thresholds.face_area_min:
        reasons.append(ReasonCode.TOO_FAR)
    elif area > thresholds.face_area_max:
        reasons.append(ReasonCode.TOO_CLOSE)

    return tuple(reasons)


class QualityGate:
    """Session-scoped OK/HOLD state machine with hysteresis (§6.4)."""

    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self._thresholds = validate_thresholds(thresholds or QualityThresholds())
        self._valid_streak = 0
        self._status = Status.HOLD

    @property
    def thresholds(self) -> QualityThresholds:
        return self._thresholds

    @property
    def status(self) -> Status:
        return self._status

    def reset(self) -> None:
        self._valid_streak = 0
        self._status = Status.HOLD

    def update(self, frame: LandmarkFrame) -> AnalysisState:
        """Advance the state machine by one analysed frame."""
        if frame.face_count == 0:
            return self._settle(Status.NO_FACE, (ReasonCode.NO_FACE,))
        if frame.face_count > 1:
            return self._settle(Status.MULTI_FACE, (ReasonCode.MULTIPLE_FACES,))

        # Leaving OK uses the outer thresholds; entering OK uses the inner ones.
        strict = self._status is not Status.OK
        reasons = evaluate_frame(frame, self._thresholds, strict=strict)
        if reasons:
            return self._settle(Status.HOLD, reasons)

        if self._status is Status.OK:
            return self._settle(Status.OK, ())

        self._valid_streak += 1
        if self._valid_streak >= self._thresholds.valid_frames_to_ok:
            return self._settle(Status.OK, ())
        return AnalysisState(
            status=Status.HOLD,
            reason=ReasonCode.UNSTABLE,
            reason_codes=(ReasonCode.UNSTABLE,),
        )

    def _settle(self, status: Status, reasons: tuple[ReasonCode, ...]) -> AnalysisState:
        if status is Status.OK:
            self._status = Status.OK
        else:
            self._status = status
            self._valid_streak = 0
        return AnalysisState(
            status=status,
            reason=reasons[0] if reasons else None,
            reason_codes=reasons,
        )
