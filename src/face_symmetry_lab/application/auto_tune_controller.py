"""When and what to auto-tune (§14.3c, §14.3d, FR-029/030).

Split out of `RealtimeLandmarkPipeline` so the "should anything change?"
policy — rolling windows and retune throttles — is separable from the frame
pipeline that applies the answers. This module decides; the pipeline acts.

Every recommendation here is display-only or cadence-only. §8.6 removed the
metric's sensitivity coefficients and part weights, so there is no longer any
scoring-affecting setting for auto-tuning to reach, and therefore no path from
here to a `config_revision`/segment/EMA reset (AC-AUTOTUNE-02).

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
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.auto_tuning import (
    recommend_analysis_interval_ms,
    recommend_brightness_gain,
    recommend_zoom_policy,
)
from face_symmetry_lab.domain.frame_adjustments import mean_luminance
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: §14.3c brightness/zoom auto-tuning (FR-029): recomputed on a slower cadence
#: than analysis, since `apply_brightness_gain` caches its lookup table per
#: exact gain value — holding gain stable across several analysed frames
#: keeps that cache hot instead of rebuilding the table each time.
DISPLAY_RETUNE_INTERVAL_SECONDS: Final[float] = 1.0

#: §14.3d cadence auto-tuning (FR-030): non-scoring, so no segment reset is needed.
#: Since §8.6 removed the metric's coefficients, cadence and display are the
#: only things left to tune — nothing here can move a measured value.
CADENCE_PROCESSING_WINDOW: Final[int] = 20
MIN_CADENCE_SAMPLES: Final[int] = 5
CADENCE_RETUNE_INTERVAL_SECONDS: Final[float] = 2.0


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
        self._processing_ms_window: deque[float] = deque(maxlen=CADENCE_PROCESSING_WINDOW)

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
