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
from face_symmetry_lab.application.ports import LandmarkProvider
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.overlay import (
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    draw_landmark_points,
)
from face_symmetry_lab.runtime.frame_scheduler import FrameScheduler

#: Consecutive failures before the UI should offer a recovery action (§14.4).
ERROR_CIRCUIT_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Immutable view of the latest analysed frame."""

    frame_id: int = 0
    status: TrackingStatus = TrackingStatus.NO_FACE
    face_count: int = 0
    landmark_count: int = 0
    dropped_frames: int = 0
    consecutive_errors: int = 0
    processing_ms: float = 0.0
    #: Present only when an analyzer is attached and the frame was scored.
    analysis: AnalysisOutcome | None = None

    @property
    def needs_recovery(self) -> bool:
        return self.consecutive_errors >= ERROR_CIRCUIT_THRESHOLD


class RealtimeLandmarkPipeline:
    """Drive one landmark detection per frame, dropping frames when busy."""

    def __init__(
        self,
        *,
        landmarker: LandmarkProvider,
        analyzer: FrameAnalyzer | None = None,
        recorder: SessionRecorder | None = None,
        scheduler: FrameScheduler | None = None,
        clock: Callable[[], float] = time.monotonic,
        point_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
        point_radius: int = DEFAULT_POINT_RADIUS,
    ) -> None:
        self._landmarker = landmarker
        self._analyzer = analyzer
        self._recorder = recorder
        self.scheduler = scheduler if scheduler is not None else FrameScheduler()
        self._clock = clock
        self._point_color = point_color
        self._point_radius = point_radius
        self._started_at = clock()
        self._frame_id = 0
        self._last_timestamp_ms = -1
        self._snapshot = PipelineSnapshot()
        self._closed = False
        self._state_lock = threading.Lock()

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
            self._record_error()
            return rgb_image, None

        with self.scheduler.slot() as acquired:
            if not acquired:
                return rgb_image, None
            return self._analyse(rgb_image)

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
            painted = draw_landmark_points(
                rgb_image,
                frame,
                color=self._point_color,
                radius=self._point_radius,
            )
        except Exception:
            # A single frame failure must not stop the stream (NFR-017).
            self._record_error(
                frame_id=self._frame_id,
                elapsed_ms=timestamp_ms,
                processing_ms=(self._clock() - started) * 1000.0,
            )
            return rgb_image, None

        analysis: AnalysisOutcome | None = None
        if self._analyzer is not None:
            try:
                analysis = self._analyzer.analyze(frame, at=started)
            except Exception:
                # Scoring must never take the preview down with it.
                analysis = None

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
            except Exception:
                # Export bookkeeping is secondary to keeping the camera alive.
                pass

        with self._state_lock:
            self._snapshot = PipelineSnapshot(
                frame_id=frame.frame_id,
                status=frame.status,
                face_count=frame.face_count,
                landmark_count=len(frame.points),
                dropped_frames=self.scheduler.dropped_frames,
                consecutive_errors=0,
                processing_ms=processing_ms,
                analysis=analysis,
            )
        return painted, analysis

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
        with self._state_lock:
            return replace(self._snapshot, dropped_frames=self.scheduler.dropped_frames)

    def reset_analysis(self) -> None:
        """Drop scores, smoothing, and hysteresis without stopping the stream."""
        if self._analyzer is not None:
            self._analyzer.reset()
        if self._recorder is not None:
            self._recorder.reset()
        self.scheduler.reset()
        with self._state_lock:
            self._snapshot = PipelineSnapshot()

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
        self._landmarker.close()
