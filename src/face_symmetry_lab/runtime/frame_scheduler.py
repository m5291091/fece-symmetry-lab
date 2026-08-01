"""Latest-frame-wins gate for the realtime pipeline (§14.3).

Capacity is one analysis per session. Frames that arrive while an analysis is
running are dropped rather than queued, so the WebRTC receive thread is never
blocked and stale frames never accumulate.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class FrameScheduler:
    """A non-blocking, capacity-1 admission gate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._dropped_frames = 0

    @property
    def dropped_frames(self) -> int:
        with self._counter_lock:
            return self._dropped_frames

    @contextmanager
    def slot(self) -> Iterator[bool]:
        """Yield ``True`` when the caller may analyse, ``False`` when it must skip."""
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            with self._counter_lock:
                self._dropped_frames += 1
            yield False
            return
        try:
            yield True
        finally:
            self._lock.release()

    def reset(self) -> None:
        with self._counter_lock:
            self._dropped_frames = 0
