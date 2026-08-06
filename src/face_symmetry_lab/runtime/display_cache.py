"""表示経路が、古くなるまで再利用してよい 1 枠だけの値。

§1.2(17): cadence 制御が解析しなかったフレームは、直前の幾何を描き直し、
直前の背景 mask を再利用してよい。ただし、それらが同じ顔についてのものだと
まだ考えられる間に限る。これは同じ問いを 2 回しているだけなので、「値と保存
した時刻」およびその TTL の規則を 2 箇所へ書き下ろすのではなく、両方がこの
枠を使う。

スレッドセーフである。解析を行う WebRTC の受信スレッドが書き、同じスレッドの
間引き経路が読む。``store``・``read``・``clear`` はそれぞれ atomic である。
domain の知識を持たず pipeline から何も import しないので、フレームも
landmarker も無しでテストできる。
"""

from __future__ import annotations

import threading


class TtlCache[T]:
    """最後に保存した値。TTL より新しい間だけ読める。"""

    __slots__ = ("_lock", "_stored_at", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None
        self._stored_at: float | None = None

    def store(self, value: T, *, at: float) -> None:
        """枠を置き換え、TTL を ``at`` から数え直す。"""
        with self._lock:
            self._value = value
            self._stored_at = at

    def read(self, *, now: float, ttl_seconds: float) -> T | None:
        """値が新しい間はそれを返し、古くなったら破棄する。

        期限切れは枠を隠すのではなく消去する。そのため、後からより広い TTL で
        読んでも、すでにフレームから外れた顔の幾何が復活することはない。
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
        """TTL に関わらず値を直ちに破棄する（検出喪失、リセット時）。"""
        with self._lock:
            self._value = None
            self._stored_at = None
