"""Single-frame analysis orchestration (§14.2 `application.analyzer`).

Sequence: quality gate → symmetry metric → smoothing. The analyzer holds no
Streamlit state, draws nothing, and never fetches a model. One instance belongs
to one session.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.quality import QualityGate, QualityThresholds
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
)
from face_symmetry_lab.domain.metrics import MetricConfig
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.pose import FacePose
from face_symmetry_lab.domain.skin_quality import (
    MEDIAPIPE_CHEEK_PAIRS,
    SkinQualityError,
    SkinQualityResult,
    compute_skin_quality,
)
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
    skin_quality: SkinQualityResult | None = None

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
        skin_pairs: tuple[tuple[int, int], ...] = MEDIAPIPE_CHEEK_PAIRS,
    ) -> None:
        self._config = config
        self._gate = QualityGate(thresholds)
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._lost_since: float | None = None
        self._skin_pairs = skin_pairs

    @property
    def config(self) -> MetricConfig:
        return self._config

    @property
    def thresholds(self) -> QualityThresholds:
        return self._gate.thresholds

    @property
    def tau_seconds(self) -> float:
        """The EMA's currently active time constant (§14.3d auto-tuning readback)."""
        return self._ema.tau_seconds

    def update(
        self,
        *,
        thresholds: QualityThresholds,
        tau_seconds: float,
        config: MetricConfig | None = None,
    ) -> None:
        """Atomically replace thresholds/tau and reset gate + EMA history (§15.2).

        Build-new-then-swap, per the Advanced settings contract: a change
        that affects scoring must not let stale gate streak or smoothed
        value leak across the change. ``config`` additionally swaps
        `part_weights`/`sensitivity` when the caller supplies an edited
        bundle (e.g. `domain.metrics.with_tunables`); omitting it keeps the
        existing bundle unchanged.
        """
        if config is not None:
            self._config = config
        self._gate = QualityGate(thresholds)
        self._ema = TimeBasedEma(tau_seconds=tau_seconds)
        self._lost_since = None

    def reset(self) -> None:
        """Drop history, smoothing, and hysteresis (§4.4, §8.8)."""
        self._gate.reset()
        self._ema.reset()
        self._lost_since = None

    def analyze(
        self,
        frame: LandmarkFrame,
        *,
        at: float,
        rgb_image: NDArray[np.uint8] | None = None,
    ) -> AnalysisOutcome:
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

        skin_quality: SkinQualityResult | None = None
        if state.status is Status.OK and rgb_image is not None:
            try:
                skin_quality = compute_skin_quality(
                    rgb_image,
                    frame,
                    pairs=self._skin_pairs,
                )
            except SkinQualityError:
                # The capture-conditioned reference is independent (§8.6.1).
                skin_quality = None

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
            skin_quality=skin_quality,
        )

    def _track_face_loss(self, status: Status, at: float) -> None:
        if status not in {Status.NO_FACE, Status.MULTI_FACE}:
            self._lost_since = None
            return
        if self._lost_since is None:
            self._lost_since = at
        elif at - self._lost_since >= FACE_LOST_RESET_SECONDS:
            self._ema.reset()
