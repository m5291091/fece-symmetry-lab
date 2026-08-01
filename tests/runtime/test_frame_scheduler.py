import threading

from face_symmetry_lab.runtime.frame_scheduler import FrameScheduler


def test_first_slot_is_acquired() -> None:
    scheduler = FrameScheduler()

    with scheduler.slot() as acquired:
        assert acquired is True

    assert scheduler.dropped_frames == 0


def test_second_concurrent_slot_is_dropped_instead_of_queued() -> None:
    scheduler = FrameScheduler()

    with scheduler.slot() as outer, scheduler.slot() as inner:
        assert outer is True
        assert inner is False

    assert scheduler.dropped_frames == 1


def test_slot_is_released_after_use() -> None:
    scheduler = FrameScheduler()

    with scheduler.slot():
        pass
    with scheduler.slot() as acquired:
        assert acquired is True


def test_slot_is_released_when_the_body_raises() -> None:
    scheduler = FrameScheduler()

    try:
        with scheduler.slot():
            raise RuntimeError("frame failed")
    except RuntimeError:
        pass

    with scheduler.slot() as acquired:
        assert acquired is True


def test_a_busy_scheduler_never_blocks_another_thread() -> None:
    scheduler = FrameScheduler()
    held = threading.Event()
    may_release = threading.Event()
    observed: list[bool] = []

    def hold() -> None:
        with scheduler.slot():
            held.set()
            may_release.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    held.wait(timeout=5)

    with scheduler.slot() as acquired:
        observed.append(acquired)

    may_release.set()
    worker.join(timeout=5)

    assert observed == [False]
    assert scheduler.dropped_frames == 1


def test_dropped_frames_counter_is_reset_with_the_session() -> None:
    scheduler = FrameScheduler()

    with scheduler.slot(), scheduler.slot():
        pass
    scheduler.reset()

    assert scheduler.dropped_frames == 0
