import math

import pytest

from face_symmetry_lab.domain.smoothing import SmoothingError, TimeBasedEma


def test_the_first_sample_is_taken_as_is() -> None:
    ema = TimeBasedEma(tau_seconds=0.4)

    assert ema.update(80.0, at=10.0) == pytest.approx(80.0)
    assert ema.value == pytest.approx(80.0)


def test_tau_seconds_reads_back_the_constructor_value() -> None:
    ema = TimeBasedEma(tau_seconds=0.7)

    assert ema.tau_seconds == pytest.approx(0.7)


def test_smoothing_moves_toward_the_new_value_without_reaching_it() -> None:
    ema = TimeBasedEma(tau_seconds=0.4)
    ema.update(80.0, at=0.0)

    smoothed = ema.update(100.0, at=0.1)

    assert 80.0 < smoothed < 100.0


def test_a_longer_gap_weights_the_new_sample_more() -> None:
    quick = TimeBasedEma(tau_seconds=0.4)
    quick.update(80.0, at=0.0)
    slow = TimeBasedEma(tau_seconds=0.4)
    slow.update(80.0, at=0.0)

    assert slow.update(100.0, at=0.35) > quick.update(100.0, at=0.05)


def test_the_time_constant_is_independent_of_frame_rate() -> None:
    """100ms 間隔の 2 標本は、10ms 間隔の 20 標本と同じ値に着地する（§8.7）。"""
    coarse = TimeBasedEma(tau_seconds=0.4)
    coarse.update(0.0, at=0.0)
    for step in (0.1, 0.2):
        coarse.update(100.0, at=step)

    fine = TimeBasedEma(tau_seconds=0.4)
    fine.update(0.0, at=0.0)
    for index in range(1, 21):
        fine.update(100.0, at=index * 0.01)

    assert coarse.value == pytest.approx(fine.value, abs=1e-9)


def test_a_long_stall_restarts_the_filter() -> None:
    ema = TimeBasedEma(tau_seconds=0.4, max_gap_seconds=2.0)
    ema.update(20.0, at=0.0)

    assert ema.update(90.0, at=5.0) == pytest.approx(90.0)


def test_time_going_backwards_restarts_the_filter() -> None:
    ema = TimeBasedEma(tau_seconds=0.4)
    ema.update(20.0, at=10.0)

    assert ema.update(90.0, at=9.0) == pytest.approx(90.0)


def test_reset_clears_the_state() -> None:
    ema = TimeBasedEma(tau_seconds=0.4)
    ema.update(20.0, at=0.0)

    ema.reset()

    assert ema.value is None
    assert ema.update(70.0, at=0.05) == pytest.approx(70.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_samples_are_refused(bad: float) -> None:
    ema = TimeBasedEma(tau_seconds=0.4)

    with pytest.raises(SmoothingError):
        ema.update(bad, at=0.0)


@pytest.mark.parametrize("tau", [0.0, -1.0, math.nan])
def test_invalid_tau_is_refused(tau: float) -> None:
    with pytest.raises(SmoothingError, match="tau"):
        TimeBasedEma(tau_seconds=tau)
