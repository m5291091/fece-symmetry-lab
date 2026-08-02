"""Pure heuristics for AI-assisted Advanced-settings auto-tuning (§14.3c/d, FR-029/030).

No new model is introduced (invariants 7, 10). Every recommendation reuses
statistics already available elsewhere in the pipeline — frame luminance,
tracked face bbox area, measured processing time, and per-part landmark
RMS distance — and clamps its output to a fixed range, so an auto-tuned
value can never exceed FR-026/FR-028/FR-030's bounds. Quality-gate
thresholds and `part_weights` are deliberately not covered here: the
former would let auto-tuning loosen the very validity bar it's measured
against, and the latter has no statistical signal to optimise against
(§14.3d, "Explain, do not judge").
"""

from __future__ import annotations

import math
from typing import Final

from face_symmetry_lab.domain.frame_adjustments import MAX_BRIGHTNESS_GAIN, MIN_BRIGHTNESS_GAIN
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: Mid-gray target for the auto-exposure loop; matches a typical 8-bit
#: camera's "correctly exposed" average.
TARGET_MEAN_LUMINANCE: float = 128.0

#: Face-area ratio at which the recommended policy equals ``TrackingZoomPolicy()``'s
#: defaults, chosen to match §8.2's face_area_min/max midpoint region.
_REFERENCE_FACE_AREA_RATIO: float = 0.12
_DEFAULT_POLICY = TrackingZoomPolicy()
_MIN_TARGET_PX = 240
_MAX_TARGET_PX = 960
_MIN_MARGIN_RATIO = 0.3
_MAX_MARGIN_RATIO = 1.2


class AutoTuningError(ValueError):
    """Invalid frame statistic supplied to an auto-tuning heuristic."""


def recommend_brightness_gain(measured_mean_luminance: float) -> float:
    """An absolute brightness gain that would put the frame at mid-gray.

    ``measured_mean_luminance`` is always a reading of the raw, uncorrected
    frame (§14.3b: gain is applied only to the landmark detector's input,
    never to what's displayed, so there is no "already-corrected" frame to
    measure), so the recommendation is a direct ``target / measured``,
    clamped to FR-028's ``[MIN_BRIGHTNESS_GAIN, MAX_BRIGHTNESS_GAIN]``.
    """
    if not math.isfinite(measured_mean_luminance) or measured_mean_luminance <= 0:
        raise AutoTuningError("measured_mean_luminance は正の有限値である必要があります。")
    gain = TARGET_MEAN_LUMINANCE / measured_mean_luminance
    return min(max(gain, MIN_BRIGHTNESS_GAIN), MAX_BRIGHTNESS_GAIN)


def recommend_zoom_policy(face_area_ratio: float) -> TrackingZoomPolicy:
    """A ``TrackingZoomPolicy`` that zooms in more the smaller the face is.

    A PC webcam frames a face far smaller than a phone selfie does (OD-05);
    this scales ``target_px``/``margin_ratio`` by the inverse square root of
    how the observed face area compares to a reference ratio, so a distant
    face gets a larger crop target and margin while a close one gets a
    smaller one. ``TrackingZoomPolicy.__post_init__`` re-validates the
    result against FR-026's bounds.
    """
    if not math.isfinite(face_area_ratio) or not (0.0 < face_area_ratio <= 1.0):
        raise AutoTuningError("face_area_ratio は 0 超 1 以下の有限値である必要があります。")
    scale_factor = math.sqrt(_REFERENCE_FACE_AREA_RATIO / face_area_ratio)
    target_px = round(
        min(max(_DEFAULT_POLICY.target_px * scale_factor, _MIN_TARGET_PX), _MAX_TARGET_PX)
    )
    margin_ratio = min(
        max(_DEFAULT_POLICY.margin_ratio * scale_factor, _MIN_MARGIN_RATIO), _MAX_MARGIN_RATIO
    )
    return TrackingZoomPolicy(
        margin_ratio=margin_ratio,
        min_window_size=_DEFAULT_POLICY.min_window_size,
        recheck_interval_ms=_DEFAULT_POLICY.recheck_interval_ms,
        target_px=target_px,
        max_scale=_DEFAULT_POLICY.max_scale,
    )


#: §15.2's own documented candidates; never scoring-relevant (FR-030).
ANALYSIS_INTERVAL_CHOICES_MS: Final[tuple[int, ...]] = (125, 200, 333)

#: Recommend the fastest choice whose interval comfortably exceeds measured
#: processing time, so the analysis loop doesn't run hotter than it can keep up with.
_INTERVAL_UTILIZATION_TARGET: Final[float] = 0.7


def recommend_analysis_interval_ms(mean_processing_ms: float) -> int:
    """The fastest `ANALYSIS_INTERVAL_CHOICES_MS` entry the system can sustain."""
    if not math.isfinite(mean_processing_ms) or mean_processing_ms < 0:
        raise AutoTuningError("mean_processing_ms は 0 以上の有限値である必要があります。")
    for interval_ms in ANALYSIS_INTERVAL_CHOICES_MS:
        if mean_processing_ms <= interval_ms * _INTERVAL_UTILIZATION_TARGET:
            return interval_ms
    return ANALYSIS_INTERVAL_CHOICES_MS[-1]


MIN_AGE_INTERVAL_MS: Final[int] = 500
MAX_AGE_INTERVAL_MS: Final[int] = 10_000
#: Matches the previous fixed defaults' ratio (125ms analysis / 1000ms age).
_AGE_INTERVAL_RATIO: Final[float] = 8.0


def recommend_age_interval_ms(analysis_interval_ms: int) -> int:
    """Keep the demographic worker's cadence proportional to the analysis one."""
    if analysis_interval_ms <= 0:
        raise AutoTuningError("analysis_interval_ms は正の整数である必要があります。")
    recommended = round(analysis_interval_ms * _AGE_INTERVAL_RATIO)
    return min(max(recommended, MIN_AGE_INTERVAL_MS), MAX_AGE_INTERVAL_MS)


MIN_HISTORY_CAPACITY: Final[int] = 100
MAX_HISTORY_CAPACITY: Final[int] = 20_000
#: How much wall-clock history the buffer should cover, regardless of cadence.
_HISTORY_TARGET_SECONDS: Final[float] = 600.0


def recommend_history_capacity(analysis_interval_ms: int) -> int:
    """A buffer size that covers a fixed time window at the given cadence."""
    if analysis_interval_ms <= 0:
        raise AutoTuningError("analysis_interval_ms は正の整数である必要があります。")
    recommended = round(_HISTORY_TARGET_SECONDS * 1000.0 / analysis_interval_ms)
    return min(max(recommended, MIN_HISTORY_CAPACITY), MAX_HISTORY_CAPACITY)


#: The §15.2 sensitivity range. Auto-tuning is the only thing that sets it
#: now, but the bound stays a domain rule rather than a caller's business.
MIN_SENSITIVITY: Final[float] = 0.1
MAX_SENSITIVITY: Final[float] = 100.0


def recommend_sensitivity(
    base_sensitivity: float, *, distance_mean: float, distance_stddev: float
) -> float:
    """Damp a part's sensitivity when its raw RMS distance is noisy.

    This is a signal-quality calibration, not a judgement of which facial
    part matters more (§14.3d): the input is the coefficient of variation of
    a single part's own raw distance samples, so a jittery detection under
    the current conditions gets a gentler score-vs-distance curve instead of
    an unstable display. A degenerate zero mean (no measurable asymmetry at
    all) is not "not enough data" and simply keeps the base value.
    """
    if not math.isfinite(base_sensitivity) or base_sensitivity <= 0:
        raise AutoTuningError("base_sensitivity は正の有限値である必要があります。")
    if not math.isfinite(distance_stddev) or distance_stddev < 0:
        raise AutoTuningError("distance_stddev は 0 以上の有限値である必要があります。")
    if not math.isfinite(distance_mean) or distance_mean < 0:
        raise AutoTuningError("distance_mean は 0 以上の有限値である必要があります。")
    if distance_mean == 0:
        return base_sensitivity
    coefficient_of_variation = distance_stddev / distance_mean
    damped = base_sensitivity / (1.0 + coefficient_of_variation)
    return min(max(damped, MIN_SENSITIVITY), MAX_SENSITIVITY)
