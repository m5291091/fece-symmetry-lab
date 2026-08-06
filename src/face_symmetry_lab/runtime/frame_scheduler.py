"""リアルタイム pipeline のための「最新フレーム優先」ゲート（§14.3）。

容量はセッションあたり解析 1 件である。解析の実行中に届いたフレームはキューへ
積まず捨てる。これにより WebRTC の受信スレッドが塞がることも、古いフレームが
溜まることもない。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class FrameScheduler:
    """非ブロッキングで容量 1 の受け入れゲート。"""

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
        """解析してよいときは ``True``、飛ばすべきときは ``False`` を yield する。"""
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
