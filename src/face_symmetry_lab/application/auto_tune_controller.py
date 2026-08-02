"""When and what to auto-tune (§14.3c, §14.3d, FR-029/030).

Split out of `RealtimeLandmarkPipeline` so the "should anything change?"
policy — rolling windows, retune throttles, and the guards that keep a
stable signal from churning the session — is separable from the frame
pipeline that applies the answers. This module decides; the pipeline acts.

Nothing here touches the landmarker, recorder, worker, or analyzer: every
method reports a recommendation, or ``None`` for "leave it alone". That
keeps AC-AUTOTUNE-01/02's failure rule easy to honour, since a swallowed
exception is simply indistinguishable from "no change".

Not thread-safe by design: every method is called from `_analyse`, which
`FrameScheduler`'s capacity-1 gate already serializes, so no lock is taken
on state only that path touches.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.auto_tuning import (
    recommend_analysis_interval_ms,
    recommend_brightness_gain,
    recommend_sensitivity,
    recommend_zoom_policy,
)
from face_symmetry_lab.domain.frame_adjustments import mean_luminance
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: §14.3c brightness/zoom auto-tuning (FR-029): recomputed on a slower cadence
#: than analysis, since `apply_brightness_gain` caches its lookup table per
#: exact gain value — holding gain stable across several analysed frames
#: keeps that cache hot instead of rebuilding the table each time.
DISPLAY_RETUNE_INTERVAL_SECONDS: Final[float] = 1.0

#: §14.3d cadence auto-tuning (FR-030): non-scoring, so no segment reset is needed.
CADENCE_PROCESSING_WINDOW: Final[int] = 20
MIN_CADENCE_SAMPLES: Final[int] = 5
CADENCE_RETUNE_INTERVAL_SECONDS: Final[float] = 2.0

#: §14.3d sensitivity auto-tuning (FR-030): scoring-affecting, throttled and
#: gated on a minimum relative change so it doesn't churn segments needlessly.
SENSITIVITY_DISTANCE_WINDOW: Final[int] = 24
MIN_SENSITIVITY_SAMPLES: Final[int] = 8
SENSITIVITY_RETUNE_INTERVAL_SECONDS: Final[float] = 10.0
SENSITIVITY_CHANGE_THRESHOLD: Final[float] = 0.05


@dataclass(frozen=True, slots=True)
class DisplayTuning:
    """§14.3c recommendations from one analysed frame. ``None`` means keep."""

    brightness_gain: float | None = None
    zoom_policy: TrackingZoomPolicy | None = None


#: Shared "nothing to do" result, so the common path allocates nothing.
_NO_DISPLAY_TUNING: Final[DisplayTuning] = DisplayTuning()


class _RetuneSchedule:
    """A monotonic "is it due yet?" gate that advances itself when it fires."""

    __slots__ = ("_interval_seconds", "_next_at")

    def __init__(self, interval_seconds: float, *, start_at: float) -> None:
        self._interval_seconds = interval_seconds
        self._next_at = start_at

    def due(self, at: float) -> bool:
        """True at most once per interval; advances the deadline when it fires."""
        if at < self._next_at:
            return False
        self._next_at = at + self._interval_seconds
        return True


class AutoTuneController:
    """Decide what §14.3c/d would like changed, from observed frame statistics."""

    def __init__(self, *, enabled: bool, started_at: float) -> None:
        self._enabled = bool(enabled)
        self._display_schedule = _RetuneSchedule(
            DISPLAY_RETUNE_INTERVAL_SECONDS, start_at=started_at
        )
        self._cadence_schedule = _RetuneSchedule(
            CADENCE_RETUNE_INTERVAL_SECONDS, start_at=started_at
        )
        self._sensitivity_schedule = _RetuneSchedule(
            SENSITIVITY_RETUNE_INTERVAL_SECONDS, start_at=started_at
        )
        self._processing_ms_window: deque[float] = deque(maxlen=CADENCE_PROCESSING_WINDOW)
        self._distance_windows: dict[str, deque[float]] = {
            name: deque(maxlen=SENSITIVITY_DISTANCE_WINDOW) for name in PART_NAMES
        }

    @property
    def enabled(self) -> bool:
        """Fixed for the session: §15.2 exposes no runtime toggle."""
        return self._enabled

    def display_tuning(
        self,
        rgb_image: NDArray[np.uint8],
        frame: LandmarkFrame,
        *,
        at: float,
    ) -> DisplayTuning:
        """§14.3c: brightness gain and zoom policy, sharing one throttle.

        Both are derived from the same frame at the same moment, so they
        share a single schedule rather than drifting onto separate ones.
        `rgb_image` must be the raw, uncorrected frame: §14.3b never
        brightens display output, so its mean luminance is a direct read of
        ambient light and the recommended gain is absolute, not relative.
        Failures are swallowed per AC-AUTOTUNE-01, leaving the value alone.
        """
        if not self._enabled or not self._display_schedule.due(at):
            return _NO_DISPLAY_TUNING

        try:
            brightness_gain: float | None = recommend_brightness_gain(mean_luminance(rgb_image))
        except Exception:
            brightness_gain = None

        zoom_policy: TrackingZoomPolicy | None = None
        # `face_area_ratio` is None unless exactly one face was confirmed,
        # which is also the only case zoom tracking is meaningful in.
        ratio = frame.face_area_ratio
        if ratio is not None:
            try:
                zoom_policy = recommend_zoom_policy(ratio)
            except Exception:
                zoom_policy = None
        return DisplayTuning(brightness_gain=brightness_gain, zoom_policy=zoom_policy)

    def cadence_tuning(
        self,
        processing_ms: float,
        *,
        at: float,
        current_interval_ms: int,
    ) -> int | None:
        """§14.3d: a new `analysis_interval_ms`, or None to keep the current one.

        Returns None when the recommendation matches what is already in
        force. `age_interval_ms` and `history_capacity` are pure functions of
        this interval, so an unchanged answer here means neither would move
        either — and skipping spares the recorder a history-buffer rebuild
        on every retune tick for the whole session (AC-AUTOTUNE-02).
        """
        if not self._enabled:
            return None
        # Sampled every analysed frame; only *read* on the throttled tick.
        self._processing_ms_window.append(processing_ms)
        if not self._cadence_schedule.due(at):
            return None
        if len(self._processing_ms_window) < MIN_CADENCE_SAMPLES:
            return None
        try:
            mean_processing_ms = statistics.fmean(self._processing_ms_window)
            recommended = recommend_analysis_interval_ms(mean_processing_ms)
        except Exception:
            return None
        return None if recommended == current_interval_ms else recommended

    def sensitivity_tuning(
        self,
        analysis: AnalysisOutcome | None,
        *,
        at: float,
        config: MetricConfig,
    ) -> Mapping[str, float] | None:
        """§14.3d: a new per-part `sensitivity` map, or None to keep the current one.

        Returns a value only when at least one part moved by more than
        `SENSITIVITY_CHANGE_THRESHOLD` — this change resets config_revision,
        segment and EMA (§15.2), so a stable signal must not trigger it.
        Parts without enough samples yet keep their current coefficient.
        `part_weights` is never recommended on (§14.3d, "Explain, do not judge").
        """
        if not self._enabled:
            return None
        self._observe_distances(analysis)
        if not self._sensitivity_schedule.due(at):
            return None

        recommended: dict[str, float] = {}
        changed = False
        for name in PART_NAMES:
            window = self._distance_windows[name]
            base = config.sensitivity[name]
            recommended[name] = base
            if len(window) < MIN_SENSITIVITY_SAMPLES:
                continue
            try:
                mean = statistics.fmean(window)
                stddev = statistics.pstdev(window, mu=mean)
                value = recommend_sensitivity(base, distance_mean=mean, distance_stddev=stddev)
            except Exception:
                continue
            recommended[name] = value
            if base > 0 and abs(value - base) / base > SENSITIVITY_CHANGE_THRESHOLD:
                changed = True
        return recommended if changed else None

    def _observe_distances(self, analysis: AnalysisOutcome | None) -> None:
        """Feed this frame's raw per-part RMS distances into the jitter windows."""
        if analysis is None or analysis.symmetry is None:
            return
        for name in PART_NAMES:
            distance = analysis.symmetry.distance(name)
            if distance is not None:
                self._distance_windows[name].append(distance)
