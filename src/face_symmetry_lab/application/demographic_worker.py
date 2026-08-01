"""Off-critical-path age/gender inference (§14.3, FR-022〜024).

Age/gender inference runs on its own background thread, decoupled from the
landmark analysis cadence via ``interval_ms``. At most one inference is ever
in flight (capacity 1): a submission that arrives too early or while busy is
dropped rather than queued, so the frame callback never blocks and stale
crops never accumulate. (`runtime.frame_scheduler.FrameScheduler` enforces
the same rule for landmark analysis, but its context-manager API cannot span
the two threads this hand-off crosses.) A result computed for a segment that
was reset before it finished is discarded on completion, so it never leaks
into a later comparison segment.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.ports import DemographicEstimator
from face_symmetry_lab.domain.models import DemographicEstimate

#: §15.2 `age_interval_ms` default. Not user-editable in this milestone.
DEFAULT_DEMOGRAPHIC_INTERVAL_MS = 1_000


class DemographicWorker:
    """Capacity-1 background age/gender estimator, gated by its own cadence."""

    def __init__(
        self,
        *,
        estimator: DemographicEstimator,
        interval_ms: int = DEFAULT_DEMOGRAPHIC_INTERVAL_MS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_ms < 0:
            raise ValueError("interval_ms は 0 以上である必要があります。")
        self._estimator = estimator
        self._interval_seconds = interval_ms / 1000.0
        self._clock = clock
        self._state_lock = threading.Lock()
        self._next_submit_at = clock()
        self._generation = 0
        self._busy = False
        self._closed = False
        self._latest: DemographicEstimate | None = None
        self._latest_version = 0

    def is_due(self) -> bool:
        """Report whether a ``submit`` right now would actually start work.

        Lets the caller skip building a crop it would only throw away: the
        frame cadence is far faster than ``interval_ms``, so most offers are
        refused. This is advisory — ``submit`` re-checks under the lock.
        """
        with self._state_lock:
            return not self._closed and not self._busy and self._clock() >= self._next_submit_at

    def submit(self, rgb_crop: NDArray[np.uint8]) -> threading.Thread | None:
        """Start a background inference if idle and due; otherwise no-op."""
        with self._state_lock:
            if self._closed or self._busy or self._clock() < self._next_submit_at:
                return None
            # Claimed under the same lock as the due check, so two callers can
            # never both pass it. Released by `_run`'s `finally`.
            self._busy = True
            self._next_submit_at = self._clock() + self._interval_seconds
            generation = self._generation
        # Detach from the caller's buffer: the pipeline may mutate or reuse
        # the source frame before this background thread reads it.
        crop = np.array(rgb_crop, copy=True)
        thread = threading.Thread(target=self._run, args=(crop, generation), daemon=True)
        try:
            thread.start()
        except Exception:
            self._release()
            raise
        return thread

    def _run(self, crop: NDArray[np.uint8], generation: int) -> None:
        try:
            estimate: DemographicEstimate | None = self._estimator.estimate(crop)
        except Exception:
            # FR-024: a failed inference must never affect symmetry analysis;
            # the previous good result (if any) simply keeps being reused.
            estimate = None
        finally:
            self._release()
        if estimate is None:
            return
        with self._state_lock:
            if self._closed or generation != self._generation:
                return
            self._latest_version += 1
            self._latest = estimate

    def _release(self) -> None:
        with self._state_lock:
            self._busy = False

    def latest(self) -> tuple[DemographicEstimate, int] | None:
        """Return the current best estimate and a version token, if any."""
        with self._state_lock:
            if self._latest is None:
                return None
            return self._latest, self._latest_version

    def reset(self) -> None:
        """Start a new segment: drop the cached result and any in-flight one."""
        with self._state_lock:
            self._generation += 1
            self._latest = None
            self._latest_version = 0
            self._next_submit_at = self._clock()

    def close(self) -> None:
        """Stop accepting submissions and drop the cached result."""
        with self._state_lock:
            self._closed = True
            self._generation += 1
            self._latest = None
