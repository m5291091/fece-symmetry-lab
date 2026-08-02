"""Realtime landmark use case.

One instance belongs to one session. ``process`` runs on the WebRTC receive
thread; ``snapshot`` is read by the Streamlit fragment. The two share only an
immutable snapshot behind a lock, and a single frame failure never stops the
stream (§14.3, §14.4, NFR-017).
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.image_input import ImageInputError, extract_face_crop
from face_symmetry_lab.application.ports import (
    DemographicSampler,
    LandmarkProvider,
    SegmentationProvider,
)
from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.auto_tuning import (
    recommend_age_interval_ms,
    recommend_analysis_interval_ms,
    recommend_brightness_gain,
    recommend_history_capacity,
    recommend_sensitivity,
    recommend_zoom_policy,
)
from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.domain.frame_adjustments import (
    BACKGROUND_COLORS,
    BackgroundMode,
    FrameAdjustmentError,
    apply_brightness_gain,
    composite_background,
    mean_luminance,
)
from face_symmetry_lab.domain.landmarks import LandmarkFrame, TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig, with_tunables
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.overlay import (
    DEFAULT_MIDLINE_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    draw_face_midline,
    draw_landmark_points,
)
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy
from face_symmetry_lab.runtime.frame_scheduler import FrameScheduler

#: Consecutive failures before the UI should offer a recovery action (§14.4).
ERROR_CIRCUIT_THRESHOLD = 3

#: Preview frames keep flowing, while landmark inference is capped at 8 Hz.
DEFAULT_ANALYSIS_INTERVAL_MS = 125
DEFAULT_GEOMETRY_TTL_MS = 250

#: §14.3c brightness/zoom auto-tuning (FR-029): recomputed on a slower cadence
#: than analysis, since `apply_brightness_gain` caches its lookup table per
#: exact gain value — holding gain stable for longer keeps that cache hot on
#: the full-video-rate `_apply_brightness` path instead of rebuilding it on
#: every analysed frame.
_DISPLAY_RETUNE_INTERVAL_SECONDS = 1.0

#: §14.3d cadence auto-tuning (FR-030): non-scoring, so no segment reset is needed.
_CADENCE_PROCESSING_WINDOW = 20
_MIN_CADENCE_SAMPLES = 5
_CADENCE_RETUNE_INTERVAL_SECONDS = 2.0

#: §14.3d sensitivity auto-tuning (FR-030): scoring-affecting, throttled and
#: gated on a minimum relative change so it doesn't churn segments needlessly.
_SENSITIVITY_DISTANCE_WINDOW = 24
_MIN_SENSITIVITY_SAMPLES = 8
_SENSITIVITY_RETUNE_INTERVAL_SECONDS = 10.0
_SENSITIVITY_CHANGE_THRESHOLD = 0.05


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Immutable view of the latest analysed frame."""

    frame_id: int = 0
    status: TrackingStatus = TrackingStatus.NO_FACE
    face_count: int = 0
    landmark_count: int = 0
    #: Frames intentionally left as preview-only by cadence control.
    throttled_frames: int = 0
    #: Frames rejected because the one-slot worker was already occupied.
    dropped_frames: int = 0
    consecutive_errors: int = 0
    processing_ms: float = 0.0
    #: Present only when an analyzer is attached and the frame was scored.
    analysis: AnalysisOutcome | None = None
    #: §14.3, FR-022〜024: produced off-cadence by a `DemographicSampler`.
    demographic_estimate: DemographicEstimate | None = None

    @property
    def needs_recovery(self) -> bool:
        return self.consecutive_errors >= ERROR_CIRCUIT_THRESHOLD


class RealtimeLandmarkPipeline:
    """Sample the latest preview frame without building a latency-prone queue."""

    def __init__(
        self,
        *,
        landmarker: LandmarkProvider,
        analyzer: FrameAnalyzer | None = None,
        recorder: SessionRecorder | None = None,
        scheduler: FrameScheduler | None = None,
        segmenter: SegmentationProvider | None = None,
        demographic_worker: DemographicSampler | None = None,
        clock: Callable[[], float] = time.monotonic,
        analysis_interval_ms: int = DEFAULT_ANALYSIS_INTERVAL_MS,
        geometry_ttl_ms: int = DEFAULT_GEOMETRY_TTL_MS,
        midline_indices: tuple[int, ...] = (),
        midline_enabled: bool = True,
        midline_color: tuple[int, int, int] = DEFAULT_MIDLINE_COLOR,
        point_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
        point_radius: int = DEFAULT_POINT_RADIUS,
        brightness_gain: float = 1.0,
        background_mode: BackgroundMode = BackgroundMode.OFF,
        auto_tune_enabled: bool = False,
    ) -> None:
        if analysis_interval_ms < 0:
            raise ValueError("analysis_interval_ms は 0 以上である必要があります。")
        if geometry_ttl_ms < 0:
            raise ValueError("geometry_ttl_ms は 0 以上である必要があります。")
        # §14.3b: validated by trying it once against a throwaway pixel,
        # reusing the same bounds `apply_brightness_gain` itself enforces.
        apply_brightness_gain(np.zeros((1, 1, 3), dtype=np.uint8), brightness_gain)
        self._landmarker = landmarker
        self._analyzer = analyzer
        self._recorder = recorder
        self._segmenter = segmenter
        self._demographic_worker = demographic_worker
        self._demographic_enabled = True
        #: (version, elapsed_ms when that version was first displayed), or
        #: None before any result. One field, so the two can never disagree.
        self._demographic_seen: tuple[int, int] | None = None
        self.scheduler = scheduler if scheduler is not None else FrameScheduler()
        self._clock = clock
        self._point_color = point_color
        self._point_radius = point_radius
        self._midline_indices = midline_indices
        self._midline_enabled = midline_enabled
        self._midline_color = midline_color
        self._brightness_gain = brightness_gain
        self._background_mode = background_mode
        self._auto_tune_enabled = bool(auto_tune_enabled)
        self._analysis_interval_seconds = analysis_interval_ms / 1000.0
        self._geometry_ttl_seconds = geometry_ttl_ms / 1000.0
        self._started_at = clock()
        self._next_analysis_at = self._started_at
        self._next_display_retune_at = self._started_at
        self._processing_ms_window: deque[float] = deque(maxlen=_CADENCE_PROCESSING_WINDOW)
        self._next_cadence_retune_at = self._started_at
        self._distance_windows: dict[str, deque[float]] = {
            name: deque(maxlen=_SENSITIVITY_DISTANCE_WINDOW) for name in PART_NAMES
        }
        self._next_sensitivity_retune_at = self._started_at
        self._throttled_frames = 0
        self._frame_id = 0
        self._last_timestamp_ms = -1
        self._snapshot = PipelineSnapshot()
        self._closed = False
        self._state_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._overlay_lock = threading.Lock()
        # Kept apart from `_overlay_lock` so that lock keeps meaning "display
        # settings and cached overlay pixels" only.
        self._demographic_lock = threading.Lock()
        self._cached_geometry: LandmarkFrame | None = None
        self._cached_geometry_at: float | None = None
        self._cached_mask: NDArray[np.float32] | None = None
        self._cached_mask_at: float | None = None

    def process(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Analyse one frame and return the image to send back to the browser."""
        painted, _ = self.process_with_analysis(rgb_image)
        return painted

    def process_with_analysis(
        self,
        rgb_image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], AnalysisOutcome | None]:
        """Return only the analysis belonging to this exact preview frame."""
        if self._closed:
            return rgb_image, None
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            self._clear_cached_geometry()
            self._clear_cached_mask()
            self._record_error()
            return rgb_image, None

        adjusted = self._apply_brightness(rgb_image)

        if not self._admit_latest_sample():
            return self._draw_cached(adjusted), None

        with self.scheduler.slot() as acquired:
            if not acquired:
                return self._draw_cached(adjusted), None
            return self._analyse(adjusted)

    def _apply_brightness(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        with self._overlay_lock:
            gain = self._brightness_gain
        if gain == 1.0:
            return rgb_image
        try:
            return apply_brightness_gain(rgb_image, gain)
        except FrameAdjustmentError:
            # gain is validated at set-time; a failure here means the frame
            # itself is unusable, so fall through and let detection fail on
            # the original image instead of masking the real problem.
            return rgb_image

    def _draw_cached(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        painted = self._draw_cached_geometry(rgb_image)
        return self._composite_cached_background(painted)

    def _admit_latest_sample(self) -> bool:
        """Admit by cadence only; skipped frames are never queued for later."""
        now = self._clock()
        with self._sample_lock:
            if now < self._next_analysis_at:
                self._throttled_frames += 1
                return False
            self._next_analysis_at = now + self._analysis_interval_seconds
            return True

    def _analyse(
        self,
        rgb_image: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], AnalysisOutcome | None]:
        started = self._clock()
        self._frame_id += 1
        timestamp_ms = max(
            int((started - self._started_at) * 1000),
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms

        try:
            frame = self._landmarker.detect_video(
                rgb_image,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
            )
            display_retune_due = self._display_retune_due(at=started)
            if frame.has_single_face:
                with self._overlay_lock:
                    self._cached_geometry = frame
                    self._cached_geometry_at = started
                self._maybe_auto_tune_zoom(frame, retune_due=display_retune_due)
            else:
                self._clear_cached_geometry()
            painted = self._paint_geometry(rgb_image, frame)
            self._maybe_auto_tune_brightness(rgb_image, retune_due=display_retune_due)
            painted = self._apply_background(rgb_image, painted, timestamp_ms, at=started)
            self._submit_demographic_sample(rgb_image, frame)
        except Exception:
            # A single frame failure must not stop the stream (NFR-017).
            self._clear_cached_geometry()
            self._record_error(
                frame_id=self._frame_id,
                elapsed_ms=timestamp_ms,
                processing_ms=(self._clock() - started) * 1000.0,
            )
            return rgb_image, None

        analysis: AnalysisOutcome | None = None
        if self._analyzer is not None:
            try:
                analysis = self._analyzer.analyze(
                    frame,
                    at=started,
                    rgb_image=rgb_image,
                )
            except Exception:
                # Scoring must never take the preview down with it.
                analysis = None

        demographic_sample = self._collect_demographic_sample(timestamp_ms)

        finished = self._clock()
        processing_ms = (finished - started) * 1000.0
        if self._recorder is not None:
            try:
                if analysis is None:
                    self._recorder.append_error(
                        frame_id=frame.frame_id,
                        elapsed_ms=timestamp_ms,
                        processing_ms=processing_ms,
                    )
                else:
                    self._recorder.append_analysis(
                        frame,
                        analysis,
                        elapsed_ms=timestamp_ms,
                        processing_ms=processing_ms,
                    )
                    if demographic_sample is not None:
                        estimate, observed_elapsed_ms = demographic_sample
                        self._recorder.attach_demographic_estimate(
                            estimate,
                            observed_elapsed_ms=observed_elapsed_ms,
                        )
            except Exception:
                # Export bookkeeping is secondary to keeping the camera alive.
                pass

        self._maybe_auto_tune_cadence(processing_ms, at=finished)
        self._maybe_auto_tune_sensitivity(analysis, at=finished)

        with self._sample_lock:
            throttled_frames = self._throttled_frames
        with self._state_lock:
            self._snapshot = PipelineSnapshot(
                frame_id=frame.frame_id,
                status=frame.status,
                face_count=frame.face_count,
                landmark_count=len(frame.points),
                throttled_frames=throttled_frames,
                dropped_frames=self.scheduler.dropped_frames,
                consecutive_errors=0,
                processing_ms=processing_ms,
                analysis=analysis,
                demographic_estimate=(
                    demographic_sample[0] if demographic_sample is not None else None
                ),
            )
        return painted, analysis

    def _paint_geometry(
        self,
        rgb_image: NDArray[np.uint8],
        frame: LandmarkFrame,
    ) -> NDArray[np.uint8]:
        painted = draw_landmark_points(
            rgb_image,
            frame,
            color=self._point_color,
            radius=self._point_radius,
        )
        with self._overlay_lock:
            midline_enabled = self._midline_enabled
        if midline_enabled:
            painted = draw_face_midline(
                painted,
                frame,
                midline_indices=self._midline_indices,
                color=self._midline_color,
            )
        return painted

    def _draw_cached_geometry(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        now = self._clock()
        with self._overlay_lock:
            frame = self._cached_geometry
            cached_at = self._cached_geometry_at
        if frame is None or cached_at is None or now - cached_at > self._geometry_ttl_seconds:
            self._clear_cached_geometry()
            return rgb_image
        return self._paint_geometry(rgb_image, frame)

    def _clear_cached_geometry(self) -> None:
        with self._overlay_lock:
            self._cached_geometry = None
            self._cached_geometry_at = None

    def _apply_background(
        self,
        source_image: NDArray[np.uint8],
        painted: NDArray[np.uint8],
        timestamp_ms: int,
        *,
        at: float,
    ) -> NDArray[np.uint8]:
        """Composite a solid background onto the display frame only (§14.3b).

        Segmentation runs on the source frame, not ``painted``, so overlay
        graphics never leak into the mask. A failure here is display-only
        and must never be mistaken for a stream failure (NFR-017): it is
        swallowed and the un-composited frame is returned.
        """
        with self._overlay_lock:
            mode = self._background_mode
            segmenter = self._segmenter
        if mode is BackgroundMode.OFF or segmenter is None:
            return painted
        try:
            mask = segmenter.segment(source_image, timestamp_ms=timestamp_ms)
            composited = composite_background(painted, mask, BACKGROUND_COLORS[mode])
        except Exception:
            return painted
        with self._overlay_lock:
            self._cached_mask = mask
            self._cached_mask_at = at
        return composited

    def _composite_cached_background(self, painted: NDArray[np.uint8]) -> NDArray[np.uint8]:
        with self._overlay_lock:
            mode = self._background_mode
            mask = self._cached_mask
            cached_at = self._cached_mask_at
        if mode is BackgroundMode.OFF or mask is None or cached_at is None:
            return painted
        now = self._clock()
        if now - cached_at > self._geometry_ttl_seconds:
            self._clear_cached_mask()
            return painted
        try:
            return composite_background(painted, mask, BACKGROUND_COLORS[mode])
        except Exception:
            return painted

    def _clear_cached_mask(self) -> None:
        with self._overlay_lock:
            self._cached_mask = None
            self._cached_mask_at = None

    def _display_retune_due(self, *, at: float) -> bool:
        """Throttle brightness/zoom recomputation to `_DISPLAY_RETUNE_INTERVAL_SECONDS`.

        `apply_brightness_gain` caches its lookup table per exact gain value
        (§14.3b); holding gain stable between retunes keeps that cache hot on
        the full-video-rate `_apply_brightness` path instead of computing a
        fresh recommendation, and rebuilding the LUT, on every analysed frame.
        """
        with self._sample_lock:
            due = at >= self._next_display_retune_at
            if due:
                self._next_display_retune_at = at + _DISPLAY_RETUNE_INTERVAL_SECONDS
        return due

    def _maybe_auto_tune_brightness(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        retune_due: bool,
    ) -> None:
        """§14.3c: nudge `brightness_gain` toward mid-gray on a slow cadence.

        Any failure to measure or compute a recommendation is swallowed and
        the current gain is kept (AC-AUTOTUNE-01) — this is a display/
        detection convenience, not part of the analysis pipeline that
        NFR-017's stream-continuity guarantee already covers.
        """
        with self._overlay_lock:
            enabled = self._auto_tune_enabled
            current_gain = self._brightness_gain
        if not enabled or not retune_due:
            return
        try:
            measured = mean_luminance(rgb_image)
            new_gain = recommend_brightness_gain(measured, current_gain=current_gain)
        except Exception:
            return
        with self._overlay_lock:
            self._brightness_gain = new_gain

    def _maybe_auto_tune_zoom(self, frame: LandmarkFrame, *, retune_due: bool) -> None:
        """§14.3c: retune the tracking-zoom policy from the tracked face area.

        No-ops when auto-tune is off, it isn't due yet, the frame lacks a
        confirmed single-face area, or the landmarker does not support zoom
        policies at all. Any failure is swallowed, keeping the previous
        policy (AC-AUTOTUNE-01).
        """
        with self._overlay_lock:
            enabled = self._auto_tune_enabled
        if not enabled or not retune_due:
            return
        ratio = frame.face_area_ratio
        if ratio is None:
            return
        update_zoom_policy = getattr(self._landmarker, "update_policy", None)
        if not callable(update_zoom_policy):
            return
        try:
            policy = recommend_zoom_policy(ratio)
        except Exception:
            return
        update_zoom_policy(policy)

    def _maybe_auto_tune_cadence(self, processing_ms: float, *, at: float) -> None:
        """§14.3d: retune cadence settings from measured processing load.

        `analysis_interval_ms`/`age_interval_ms`/`history_capacity` don't
        affect scoring (§15.2), so this never touches config_revision,
        segment, or EMA — only `_set_analysis_interval_ms`, the demographic
        worker's own interval, and the recorder's non-disruptive resize.
        """
        with self._overlay_lock:
            enabled = self._auto_tune_enabled
        if not enabled:
            return
        self._processing_ms_window.append(processing_ms)
        with self._sample_lock:
            due = at >= self._next_cadence_retune_at
            if due:
                self._next_cadence_retune_at = at + _CADENCE_RETUNE_INTERVAL_SECONDS
        if not due or len(self._processing_ms_window) < _MIN_CADENCE_SAMPLES:
            return
        try:
            mean_processing_ms = statistics.fmean(self._processing_ms_window)
            new_interval_ms = recommend_analysis_interval_ms(mean_processing_ms)
        except Exception:
            return
        self._set_analysis_interval_ms(new_interval_ms)
        if self._demographic_worker is not None:
            try:
                age_interval_ms = recommend_age_interval_ms(new_interval_ms)
            except Exception:
                pass
            else:
                self._demographic_worker.set_interval_ms(age_interval_ms)
        if self._recorder is not None:
            try:
                capacity = recommend_history_capacity(new_interval_ms)
            except Exception:
                pass
            else:
                self._recorder.resize_history_capacity(capacity)

    def _maybe_auto_tune_sensitivity(
        self,
        analysis: AnalysisOutcome | None,
        *,
        at: float,
    ) -> None:
        """§14.3d: retune `sensitivity` from each part's raw-distance jitter.

        `part_weights` is never touched (§14.3d, "Explain, do not judge"). A
        change is applied — via the same `apply_advanced_settings` path a
        manual edit uses, so config_revision/segment/EMA reset exactly as
        §15.2 already requires for a scoring-affecting change — only when at
        least one part's recommendation differs from its current value by
        more than `_SENSITIVITY_CHANGE_THRESHOLD`, so a stable signal never
        churns the session.
        """
        with self._overlay_lock:
            enabled = self._auto_tune_enabled
        if not enabled or self._analyzer is None:
            return
        if analysis is not None and analysis.symmetry is not None:
            for name in PART_NAMES:
                distance = analysis.symmetry.distance(name)
                if distance is not None:
                    self._distance_windows[name].append(distance)
        if at < self._next_sensitivity_retune_at:
            return
        self._next_sensitivity_retune_at = at + _SENSITIVITY_RETUNE_INTERVAL_SECONDS

        config = self._analyzer.config
        recommended: dict[str, float] = {}
        changed = False
        for name in PART_NAMES:
            window = self._distance_windows[name]
            base = config.sensitivity[name]
            if len(window) < _MIN_SENSITIVITY_SAMPLES:
                recommended[name] = base
                continue
            try:
                mean = statistics.fmean(window)
                stddev = statistics.pstdev(window, mu=mean)
                value = recommend_sensitivity(base, distance_mean=mean, distance_stddev=stddev)
            except Exception:
                value = base
            recommended[name] = value
            if base > 0 and abs(value - base) / base > _SENSITIVITY_CHANGE_THRESHOLD:
                changed = True
        if not changed:
            return
        try:
            new_config = with_tunables(config, weights=config.weights, sensitivity=recommended)
            self.apply_advanced_settings(
                thresholds=self._analyzer.thresholds,
                tau_seconds=self._analyzer.tau_seconds,
                zoom_policy=self.zoom_policy or TrackingZoomPolicy(),
                metric_config=new_config,
            )
        except Exception:
            return

    def _active_demographic_worker(self) -> DemographicSampler | None:
        """Return the worker only while the user has age/gender display on."""
        if self._demographic_worker is None:
            return None
        with self._demographic_lock:
            return self._demographic_worker if self._demographic_enabled else None

    def _submit_demographic_sample(
        self,
        rgb_image: NDArray[np.uint8],
        frame: LandmarkFrame,
    ) -> None:
        """Hand a single-face crop to the off-critical-path worker (§14.3).

        Age/gender inference never blocks this WebRTC callback thread. The
        worker's cadence is far slower than the analysis cadence, so ask it
        first — otherwise most frames would pay for a crop it discards.
        """
        worker = self._active_demographic_worker()
        if worker is None or not frame.has_single_face or not worker.is_due():
            return
        try:
            crop = extract_face_crop(rgb_image, frame)
        except ImageInputError:
            return
        worker.submit(crop)

    def _collect_demographic_sample(
        self,
        timestamp_ms: int,
    ) -> tuple[DemographicEstimate, int] | None:
        """Read the worker's latest result, marking reuse across frames.

        `demographic_is_cached` (FR-023) distinguishes a fresh inference
        (this exact frame) from a redisplay of the same segment's last
        result, since the worker runs far less often than analysis frames.
        """
        worker = self._active_demographic_worker()
        if worker is None:
            return None
        latest = worker.latest()
        if latest is None:
            return None
        estimate, version = latest
        with self._demographic_lock:
            seen = self._demographic_seen
            if seen is None or seen[0] != version:
                self._demographic_seen = (version, timestamp_ms)
                return estimate, timestamp_ms
            observed_elapsed_ms = seen[1]
        return estimate.model_copy(update={"is_cached": True}), observed_elapsed_ms

    def _record_error(
        self,
        *,
        frame_id: int | None = None,
        elapsed_ms: int | None = None,
        processing_ms: float | None = None,
    ) -> None:
        with self._state_lock:
            self._snapshot = replace(
                self._snapshot,
                status=TrackingStatus.ERROR,
                face_count=0,
                landmark_count=0,
                consecutive_errors=self._snapshot.consecutive_errors + 1,
                analysis=None,
            )
        if (
            self._recorder is not None
            and frame_id is not None
            and elapsed_ms is not None
            and processing_ms is not None
        ):
            with suppress(Exception):
                self._recorder.append_error(
                    frame_id=frame_id,
                    elapsed_ms=elapsed_ms,
                    processing_ms=processing_ms,
                )

    def snapshot(self) -> PipelineSnapshot:
        """Return the latest immutable result for the UI thread."""
        with self._sample_lock:
            throttled_frames = self._throttled_frames
        with self._state_lock:
            return replace(
                self._snapshot,
                throttled_frames=throttled_frames,
                dropped_frames=self.scheduler.dropped_frames,
            )

    def reset_analysis(self) -> None:
        """Drop scores, smoothing, and hysteresis without stopping the stream."""
        if self._analyzer is not None:
            self._analyzer.reset()
        if self._recorder is not None:
            self._recorder.reset()
        self.scheduler.reset()
        self._clear_cached_geometry()
        self._clear_cached_mask()
        self._reset_demographic_worker()
        with self._sample_lock:
            self._next_analysis_at = self._clock()
            self._throttled_frames = 0
        with self._state_lock:
            self._snapshot = PipelineSnapshot()

    def _reset_demographic_worker(self) -> None:
        """New comparison segment: drop the cached estimate (§8.8, FR-023)."""
        if self._demographic_worker is not None:
            self._demographic_worker.reset()
        with self._demographic_lock:
            self._demographic_seen = None

    def apply_advanced_settings(
        self,
        *,
        thresholds: QualityThresholds,
        tau_seconds: float,
        zoom_policy: TrackingZoomPolicy,
        metric_config: MetricConfig | None = None,
        history_capacity: int | None = None,
        age_interval_ms: int | None = None,
        analysis_interval_ms: int | None = None,
    ) -> None:
        """Atomically apply new Advanced settings (§15.2).

        Reaches every component whose behaviour depends on these values:
        the quality gate and EMA are rebuilt (dropping stale streak/smoothed
        state), the tracking-zoom policy is swapped in place against the
        still-valid last known face, and the recorder starts a new segment
        with a recomputed config fingerprint so exported records before and
        after this change are never conflated. ``metric_config`` additionally
        swaps `part_weights`/`sensitivity` (e.g. via
        `domain.metrics.with_tunables`); ``history_capacity`` resizes the
        session buffer; ``age_interval_ms`` and ``analysis_interval_ms``
        retune the two independent sampling cadences. Any left as ``None``
        keeps its current value.
        """
        if self._analyzer is not None:
            self._analyzer.update(
                thresholds=thresholds,
                tau_seconds=tau_seconds,
                config=metric_config,
            )
        update_zoom_policy = getattr(self._landmarker, "update_policy", None)
        if callable(update_zoom_policy):
            update_zoom_policy(zoom_policy)
        if self._recorder is not None:
            self._recorder.reset(
                thresholds=thresholds,
                tau_seconds=tau_seconds,
                metric_config=metric_config,
                history_capacity=history_capacity,
                analysis_interval_ms=analysis_interval_ms,
            )
        if analysis_interval_ms is not None:
            self._set_analysis_interval_ms(analysis_interval_ms)
        if age_interval_ms is not None and self._demographic_worker is not None:
            self._demographic_worker.set_interval_ms(age_interval_ms)
        self._clear_cached_geometry()
        self._clear_cached_mask()
        self._reset_demographic_worker()

    def _set_analysis_interval_ms(self, interval_ms: int) -> None:
        """Retune the landmark-analysis cadence (§15.2 `analysis_interval_ms`).

        Independent of the WebRTC frame rate and of `age_interval_ms`. Takes
        effect immediately so a throttled frame right after this call is not
        stuck waiting out the previous cadence.
        """
        if interval_ms < 0:
            raise ValueError("analysis_interval_ms は 0 以上である必要があります。")
        with self._sample_lock:
            self._analysis_interval_seconds = interval_ms / 1000.0
            self._next_analysis_at = self._clock()

    def set_midline_enabled(self, enabled: bool) -> None:
        """Update the display-only centre-line preference thread-safely."""
        with self._overlay_lock:
            self._midline_enabled = bool(enabled)

    def set_brightness_gain(self, gain: float) -> None:
        """Update the brightness gain applied to detection input and display."""
        apply_brightness_gain(np.zeros((1, 1, 3), dtype=np.uint8), gain)
        with self._overlay_lock:
            self._brightness_gain = gain

    def set_background_mode(self, mode: BackgroundMode) -> None:
        """Update the display-only background replacement thread-safely."""
        with self._overlay_lock:
            self._background_mode = mode

    def set_auto_tune_enabled(self, enabled: bool) -> None:
        """Toggle AI-assisted brightness/zoom auto-tuning (§14.3c, FR-029).

        While on, `brightness_gain` and the landmarker's zoom policy are
        recomputed from frame statistics each analysed frame instead of
        following manual UI input; turning it off freezes whatever values
        auto-tuning last picked for manual editing to resume from.
        """
        with self._overlay_lock:
            self._auto_tune_enabled = bool(enabled)

    @property
    def brightness_gain(self) -> float:
        """The gain currently applied to detection input and display (§15.2)."""
        with self._overlay_lock:
            return self._brightness_gain

    @property
    def zoom_policy(self) -> TrackingZoomPolicy | None:
        """The landmarker's active zoom policy, or None if it doesn't track one."""
        return getattr(self._landmarker, "policy", None)

    @property
    def analysis_interval_ms(self) -> int:
        """The currently active analysis cadence (§15.2, §14.3d auto-tuning readback)."""
        with self._sample_lock:
            return round(self._analysis_interval_seconds * 1000)

    @property
    def metric_config(self) -> MetricConfig | None:
        """The analyzer's current bundle, or None when no analyzer is attached.

        Lets a caller (e.g. the Advanced settings UI) derive an edited
        `part_weights`/`sensitivity` bundle via `domain.metrics.with_tunables`
        from whatever is live right now, including a bundle from a previous
        `apply_advanced_settings` call.
        """
        return self._analyzer.config if self._analyzer is not None else None

    @property
    def has_demographic_worker(self) -> bool:
        """Whether age/gender inference is available for this session.

        Fixed when the pipeline is built, so the UI can gate its toggle on
        the worker that actually exists rather than re-resolving the model
        and disagreeing with it.
        """
        return self._demographic_worker is not None

    def set_demographic_enabled(self, enabled: bool) -> None:
        """Toggle age/gender inference (§15.1, default ON) thread-safely."""
        with self._demographic_lock:
            self._demographic_enabled = bool(enabled)

    def export_bundle(self) -> SessionExport | None:
        """Return an immutable export snapshot without exposing live history."""
        if self._recorder is None:
            return None
        return self._recorder.export_bundle()

    def close(self) -> None:
        """Release the landmark runtime and stop accepting frames."""
        if self._closed:
            return
        self._closed = True
        self._clear_cached_geometry()
        self._clear_cached_mask()
        self._landmarker.close()
        if self._segmenter is not None:
            self._segmenter.close()
        if self._demographic_worker is not None:
            self._demographic_worker.close()
