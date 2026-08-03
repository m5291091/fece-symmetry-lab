"""A single-slot value that a display path may reuse until it goes stale.

§1.2(17): a frame that cadence control left un-analysed may keep redrawing
the last geometry and reuse the last background mask, but only while both
are still plausibly about the same face. That is one question asked twice,
so both use this slot rather than open-coding "value plus when it was
stored" — and its TTL rules — in two places.

Thread-safe: written from the WebRTC receive thread that runs the analysis
and read from the same thread's throttled path, with ``store``, ``read``
and ``clear`` each atomic. Holds no domain knowledge and imports nothing
from the pipeline, so it is testable without a frame or a landmarker.
"""

from __future__ import annotations

import threading


class TtlCache[T]:
    """The latest stored value, readable only while younger than a TTL."""

    __slots__ = ("_lock", "_stored_at", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None
        self._stored_at: float | None = None

    def store(self, value: T, *, at: float) -> None:
        """Replace the slot, restarting the TTL from ``at``."""
        with self._lock:
            self._value = value
            self._stored_at = at

    def read(self, *, now: float, ttl_seconds: float) -> T | None:
        """Return the value while it is fresh, dropping it once it is not.

        Expiry clears the slot rather than merely hiding it, so a later read
        with a wider TTL can never resurrect geometry belonging to a face
        that has since left the frame.
        """
        with self._lock:
            value = self._value
            stored_at = self._stored_at
            if value is None or stored_at is None or now - stored_at > ttl_seconds:
                self._value = None
                self._stored_at = None
                return None
            return value

    def clear(self) -> None:
        """Drop the value immediately, TTL or not (detection loss, reset)."""
        with self._lock:
            self._value = None
            self._stored_at = None
