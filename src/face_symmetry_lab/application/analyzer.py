"""Single-frame analysis orchestration (§14.2 `application.analyzer`).

Sequence: quality gate → symmetry metric → smoothing. The analyzer holds no
Streamlit state, draws nothing, and never fetches a model. One instance belongs
to one session.
"""

from __future__ import annotations

from dataclasses import dataclass

from face_symmetry_lab.application.quality import QualityGate, QualityThresholds
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
)
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS, TimeBasedEma
from face_symmetry_lab.domain.symmetry import (
    SymmetryError,
    SymmetryResult,
    compute_symmetry,
)

#: §8.8 resets the filter once the face has been absent for this long.
FACE_LOST_RESET_SECONDS: float = 0.5


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Immutable result of analysing one frame."""

    frame_id: int
    state: AnalysisState
    symmetry: SymmetryResult | None
    overall_smooth: float | None
    pose: FacePose | None
    face_area_ratio: float | None
    metric_version: str
    landmark_map_version: str
    sensitivity_frozen: bool

    @property
    def overall_raw(self) -> float | None:
        return self.symmetry.overall if self.symmetry is not None else None


class FrameAnalyzer:
    """Turn one landmark frame into a state, a score, and a smoothed value."""

    def __init__(
        self,
        *,
        config: MetricConfig,
        thresholds: QualityThresholds | None = None,
        tau_seconds: float = DEFAULT_TAU_SECONDS,
    ) -> None:
        self._config = config
        self._gate = QualityGate(thresholds)
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._lost_since: float | None = None

    @property
    def config(self) -> MetricConfig:
        return self._config

    @property
    def thresholds(self) -> QualityThresholds:
        return self._gate.thresholds

    def reset(self) -> None:
        """Drop history, smoothing, and hysteresis (§4.4, §8.8)."""
        self._gate.reset()
        self._ema.reset()
        self._lost_since = None

    def analyze(self, frame: LandmarkFrame, *, at: float) -> AnalysisOutcome:
        state = self._gate.update(frame)
        self._track_face_loss(state.status, at)

        symmetry: SymmetryResult | None = None
        if state.status is Status.OK:
            try:
                symmetry = compute_symmetry(to_pixel_array(frame), self._config)
            except (SymmetryError, LandmarkError):
                # A metric failure is this frame's failure, not the stream's.
                self._gate.reset()
                self._ema.reset()
                state = AnalysisState(
                    status=Status.ERROR,
                    reason=ReasonCode.LOW_QUALITY,
                    reason_codes=(ReasonCode.LOW_QUALITY,),
                )
                symmetry = None

        smooth: float | None = None
        if symmetry is not None and symmetry.overall is not None:
            smooth = self._ema.update(symmetry.overall, at=at)

        return AnalysisOutcome(
            frame_id=frame.frame_id,
            state=state,
            symmetry=symmetry,
            overall_smooth=smooth,
            pose=frame.pose,
            face_area_ratio=frame.face_area_ratio,
            metric_version=self._config.metric_version,
            landmark_map_version=self._config.landmark_map_version,
            sensitivity_frozen=self._config.sensitivity_frozen,
        )

    def _track_face_loss(self, status: Status, at: float) -> None:
        if status not in {Status.NO_FACE, Status.MULTI_FACE}:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = at
        elif at - self._lost_since >= FACE_LOST_RESET_SECONDS:
            self._ema.reset()
