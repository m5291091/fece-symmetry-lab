import threading

from face_symmetry_lab.runtime.display_cache import TtlCache


def test_a_fresh_value_is_returned_until_the_ttl_elapses() -> None:
    cache: TtlCache[str] = TtlCache()
    cache.store("geometry", at=10.0)

    assert cache.read(now=10.0, ttl_seconds=0.25) == "geometry"
    assert cache.read(now=10.25, ttl_seconds=0.25) == "geometry"


def test_an_expired_value_is_dropped_rather_than_returned_once_more() -> None:
    """§1.2(17): 古いオーバーレイを、後の顔へ描き直してはならない。"""
    cache: TtlCache[str] = TtlCache()
    cache.store("geometry", at=10.0)

    assert cache.read(now=10.26, ttl_seconds=0.25) is None
    # 隠すのではなく破棄する。TTL を広げても復活させられない。
    assert cache.read(now=10.26, ttl_seconds=100.0) is None


def test_an_empty_cache_reads_as_none() -> None:
    cache: TtlCache[str] = TtlCache()

    assert cache.read(now=0.0, ttl_seconds=1.0) is None


def test_storing_again_restarts_the_ttl() -> None:
    cache: TtlCache[str] = TtlCache()
    cache.store("first", at=10.0)
    cache.store("second", at=10.2)

    assert cache.read(now=10.4, ttl_seconds=0.25) == "second"


def test_clear_drops_a_value_that_is_still_within_its_ttl() -> None:
    """検出失敗・複数顔・リセットは、いずれも即座に反映されなければならない。"""
    cache: TtlCache[str] = TtlCache()
    cache.store("geometry", at=10.0)

    cache.clear()

    assert cache.read(now=10.0, ttl_seconds=0.25) is None


def test_concurrent_store_and_read_never_observe_a_torn_slot() -> None:
    """WebRTC の受信スレッドが書き、間引き経路が読む。"""
    cache: TtlCache[tuple[str, float]] = TtlCache()
    observed: list[tuple[str, float] | None] = []
    stop = threading.Event()

    def writer() -> None:
        for index in range(2_000):
            cache.store(("value", float(index)), at=0.0)
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            observed.append(cache.read(now=0.0, ttl_seconds=1.0))

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(value is None or value[0] == "value" for value in observed)
