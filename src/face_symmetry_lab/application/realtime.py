"""Realtime landmark use case.

One instance belongs to one session. ``process`` runs on the WebRTC receive
thread; ``snapshot`` is read by the Streamlit fragment. The two share only an
immutable snapshot behind a lock, and a single frame failure never stops the
stream (§14.3, §14.4, NFR-017).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.analyzer import AnalysisOutcome, FrameAnalyzer
from face_symmetry_lab.application.auto_tune_controller import AutoTuneController, DisplayTuning
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
    recommend_history_capacity,
)
from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.domain.frame_adjustments import (
    BACKGROUND_COLORS,
    BackgroundMode,
    FrameAdjustmentError,
    apply_brightness_gain,
    composite_background,
)
from face_symmetry_lab.domain.landmarks import LandmarkFrame, TrackingStatus
from face_symmetry_lab.domain.metrics import MetricConfig, with_tunables
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

#: FR-013: the effective geometry/mask cache TTL never drops below this floor,
#: but is widened to `analysis_interval_ms * this` whenever cadence auto-tuning
#: (§14.3d) picks a slower interval — a TTL shorter than the sampling interval
#: would expire the cache every single cycle, which is exactly what caused the
#: reported background-replacement flicker at the 333ms candidate.
_GEOMETRY_TTL_SAFETY_MULTIPLIER = 2.0


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
    #: §14.3f: a NFR-002 diagnostic breakdown of `processing_ms`, not an
    #: exhaustive accounting (overlay drawing, auto-tune bookkeeping, and
    #: recorder writes are not included). 0.0 when that stage did not run
    #: this frame (e.g. `segmentation_ms` with background replacement off).
    landmark_ms: float = 0.0
    segmentation_ms: float = 0.0
    scoring_ms: float = 0.0
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
        self._analysis_interval_seconds = analysis_interval_ms / 1000.0
        #: Floor for the effective TTL widened by `_recompute_geometry_ttl`,
        #: which only cadence auto-tuning calls (§14.3d) — an explicitly
        #: configured `geometry_ttl_ms`/`analysis_interval_ms` pair (e.g. from
        #: `apply_advanced_settings` or a test) is respected as-is otherwise.
        self._base_geometry_ttl_seconds = geometry_ttl_ms / 1000.0
        self._geometry_ttl_seconds = self._base_geometry_ttl_seconds
        self._started_at = clock()
        self._next_analysis_at = self._started_at
        #: §14.3c/d: owns every retune window and throttle. Only ever touched
        #: from `_analyse`, which `FrameScheduler` serializes, so it needs no
        #: lock of its own.
        self._auto_tune = AutoTuneController(
            enabled=auto_tune_enabled,
            started_at=self._started_at,
        )
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
        """Return only the analysis belonging to this exact preview frame.

        §14.3b: brightness gain is applied only to the frame handed to the
        landmark detector (inside `_analyse`), never to what's returned here
        — the user always sees the raw camera feed, and the AI's detection
        aid stays invisible. This also means `apply_brightness_gain` only
        runs at the analysis cadence (~8Hz) instead of on every video frame.
        """
        if self._closed:
            return rgb_image, None
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            self._clear_cached_geometry()
            self._clear_cached_mask()
            self._record_error()
            return rgb_image, None

        if not self._admit_latest_sample():
            return self._draw_cached(rgb_image), None

        with self.scheduler.slot() as acquired:
            if not acquired:
                return self._draw_cached(rgb_image), None
            return self._analyse(rgb_image)

    def _apply_brightness(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Brighten a copy for the landmark detector only (§14.3b)."""
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

        landmark_ms = 0.0
        segmentation_ms = 0.0
        try:
            # §14.3b: brightened only for the detector; `rgb_image` (raw)
            # stays what gets painted on and returned for display below.
            detection_input = self._apply_brightness(rgb_image)
            frame = self._landmarker.detect_video(
                detection_input,
                frame_id=self._frame_id,
                timestamp_ms=timestamp_ms,
            )
            landmark_ms = (self._clock() - started) * 1000.0
            if frame.has_single_face:
                with self._overlay_lock:
                    self._cached_geometry = frame
                    self._cached_geometry_at = started
            else:
                self._clear_cached_geometry()
            # §14.3c: measured on the raw frame, and only ever affecting the
            # *next* one, so where this sits relative to painting is free.
            self._apply_display_tuning(self._auto_tune.display_tuning(rgb_image, frame, at=started))
            painted = self._paint_geometry(rgb_image, frame)
            painted, segmentation_ms = self._apply_background(
                rgb_image, painted, timestamp_ms, at=started
            )
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

        scoring_ms = 0.0
        analysis: AnalysisOutcome | None = None
        if self._analyzer is not None:
            scoring_started = self._clock()
            try:
                analysis = self._analyzer.analyze(
                    frame,
                    at=started,
                    rgb_image=rgb_image,
                )
            except Exception:
                # Scoring must never take the preview down with it.
                analysis = None
            scoring_ms = (self._clock() - scoring_started) * 1000.0

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

        self._apply_cadence_tuning(
            self._auto_tune.cadence_tuning(
                processing_ms,
                at=finished,
                current_interval_ms=self.analysis_interval_ms,
            )
        )
        self._apply_sensitivity_tuning(analysis, at=finished)

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
                landmark_ms=landmark_ms,
                segmentation_ms=segmentation_ms,
                scoring_ms=scoring_ms,
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
    ) -> tuple[NDArray[np.uint8], float]:
        """Composite a solid background onto the display frame only (§14.3b).

        Returns the frame and how long segmentation took (§14.3f), which is
        ``0.0`` whenever replacement is off or unavailable and nothing ran.

        Segmentation runs on the source frame, not ``painted``, so overlay
        graphics never leak into the mask. A failure here is display-only
        and must never be mistaken for a stream failure (NFR-017): it is
        swallowed and the un-composited frame is returned.
        """
        with self._overlay_lock:
            mode = self._background_mode
            segmenter = self._segmenter
        if mode is BackgroundMode.OFF or segmenter is None:
            return painted, 0.0
        started = self._clock()
        try:
            mask = segmenter.segment(source_image, timestamp_ms=timestamp_ms)
            composited = composite_background(painted, mask, BACKGROUND_COLORS[mode])
        except Exception:
            return painted, (self._clock() - started) * 1000.0
        with self._overlay_lock:
            self._cached_mask = mask
            self._cached_mask_at = at
        return composited, (self._clock() - started) * 1000.0

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

    def _apply_display_tuning(self, tuning: DisplayTuning) -> None:
        """Adopt §14.3c's brightness/zoom recommendations, if it made any."""
        if tuning.brightness_gain is not None:
            with self._overlay_lock:
                self._brightness_gain = tuning.brightness_gain
        if tuning.zoom_policy is not None:
            self._update_zoom_policy(tuning.zoom_policy)

    def _update_zoom_policy(self, policy: TrackingZoomPolicy) -> None:
        """Push a zoom policy to the landmarker, if it tracks one at all.

        `LandmarkProvider` does not require zoom support (§14.3a wraps a
        plain provider), so a landmarker without `update_policy` is a
        no-op rather than an error.
        """
        update_zoom_policy = getattr(self._landmarker, "update_policy", None)
        if callable(update_zoom_policy):
            update_zoom_policy(policy)

    def _apply_cadence_tuning(self, new_interval_ms: int | None) -> None:
        """Adopt §14.3d's cadence recommendation, if it made one.

        `analysis_interval_ms`/`age_interval_ms`/`history_capacity` don't
        affect scoring (§15.2), so this never touches config_revision,
        segment, or EMA — only the sampling cadence, the demographic
        worker's own interval, and the recorder's non-disruptive resize.
        """
        if new_interval_ms is None:
            return
        self._set_analysis_interval_ms(new_interval_ms)
        self._recompute_geometry_ttl()
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

    def _apply_sensitivity_tuning(self, analysis: AnalysisOutcome | None, *, at: float) -> None:
        """Adopt §14.3d's `sensitivity` recommendation, if it made one.

        Goes through `apply_advanced_settings` so config_revision, segment
        and EMA reset exactly as §15.2 requires of any scoring-affecting
        change. `part_weights` is passed through untouched (§14.3d,
        "Explain, do not judge").
        """
        # Checked before touching the analyzer at all: with auto-tuning off
        # nothing here may read its config, let alone swap it.
        if not self._auto_tune.enabled or self._analyzer is None:
            return
        config = self._analyzer.config
        recommended = self._auto_tune.sensitivity_tuning(analysis, at=at, config=config)
        if recommended is None:
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
        self._update_zoom_policy(zoom_policy)
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

    def _recompute_geometry_ttl(self) -> None:
        """FR-013: widen the geometry/mask cache TTL to track auto-tuned cadence.

        Called only from `_maybe_auto_tune_cadence` — a TTL shorter than the
        sampling interval would expire the cache every single cycle, which is
        what caused the reported background-replacement flicker once cadence
        auto-tuning picked the 333ms candidate. Never called for an explicitly
        configured `analysis_interval_ms` (constructor or `apply_advanced_settings`),
        whose caller-provided `geometry_ttl_ms` is respected as-is.
        """
        self._geometry_ttl_seconds = max(
            self._base_geometry_ttl_seconds,
            self._analysis_interval_seconds * _GEOMETRY_TTL_SAFETY_MULTIPLIER,
        )

    def set_midline_enabled(self, enabled: bool) -> None:
        """Update the display-only centre-line preference thread-safely."""
        with self._overlay_lock:
            self._midline_enabled = bool(enabled)

    def set_background_mode(self, mode: BackgroundMode) -> None:
        """Update the display-only background replacement thread-safely."""
        with self._overlay_lock:
            self._background_mode = mode

    @property
    def brightness_gain(self) -> float:
        """The gain currently applied to the detector's input only (§14.3b).

        Never applied to display output, so this is a readback of the AI's
        invisible detection aid rather than of anything the user can see.
        """
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
    def geometry_ttl_ms(self) -> int:
        """The currently active geometry/mask cache TTL (FR-013 readback).

        Widened by cadence auto-tuning (§14.3d) to always exceed
        `analysis_interval_ms`, so a throttled frame between two analysed
        ones never finds the cache already expired.
        """
        return round(self._geometry_ttl_seconds * 1000)

    @property
    def metric_config(self) -> MetricConfig | None:
        """The analyzer's live bundle, or None when no analyzer is attached.

        A readback of whichever bundle is in force right now, including one
        that sensitivity auto-tuning (§14.3d) derived via
        `domain.metrics.with_tunables` and swapped in.
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
