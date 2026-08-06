"""解析周期と表示 TTL 単体のテスト（§1.2(17), §14.3d, §15.2）。"""

import pytest

from face_symmetry_lab.application.analysis_cadence import (
    DEFAULT_ANALYSIS_INTERVAL_MS,
    DEFAULT_DISPLAY_TTL_MS,
    AnalysisCadence,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _cadence(clock: FakeClock, **overrides) -> AnalysisCadence:
    settings = {
        "interval_ms": DEFAULT_ANALYSIS_INTERVAL_MS,
        "display_ttl_ms": DEFAULT_DISPLAY_TTL_MS,
        "clock": clock,
        "started_at": clock(),
    }
    settings.update(overrides)
    return AnalysisCadence(**settings)


def test_a_negative_interval_is_rejected() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError):
        _cadence(clock, interval_ms=-1)


def test_a_negative_display_ttl_is_rejected() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError):
        _cadence(clock, display_ttl_ms=-1)


def test_the_first_frame_is_admitted() -> None:
    clock = FakeClock()
    cadence = _cadence(clock)

    assert cadence.admit() is True


def test_a_frame_inside_the_interval_is_throttled_rather_than_queued() -> None:
    """間引いたフレームは後回しにも積まない。最新のものだけを見る。"""
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=100)
    cadence.admit()

    clock.advance(0.01)

    assert cadence.admit() is False
    assert cadence.throttled_frames == 1


def test_a_frame_after_the_interval_is_admitted_again() -> None:
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=100)
    cadence.admit()

    clock.advance(0.2)

    assert cadence.admit() is True


def test_a_zero_interval_admits_every_frame() -> None:
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=0)

    assert [cadence.admit() for _ in range(3)] == [True, True, True]


def test_retuning_the_interval_takes_effect_immediately() -> None:
    """直後の間引き判定が古い周期を待たされてはならない。"""
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=1000)
    cadence.admit()
    assert cadence.admit() is False

    cadence.set_interval_ms(0)

    assert cadence.interval_ms == 0
    assert cadence.admit() is True


def test_retuning_to_a_negative_interval_is_rejected() -> None:
    clock = FakeClock()
    cadence = _cadence(clock)

    with pytest.raises(ValueError):
        cadence.set_interval_ms(-1)


def test_the_display_ttl_starts_at_its_configured_floor() -> None:
    clock = FakeClock()
    cadence = _cadence(clock, display_ttl_ms=250)

    assert cadence.display_ttl_ms == 250


def test_widening_keeps_the_ttl_above_the_sampling_interval() -> None:
    """§1.2(17): 間隔より短い TTL は毎周期キャッシュを失効させ、333ms 候補で
    報告された背景置換のちらつきを起こした。"""
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=333, display_ttl_ms=250)

    cadence.widen_display_ttl()

    assert cadence.display_ttl_ms > 333


def test_widening_never_drops_below_the_configured_floor() -> None:
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=10, display_ttl_ms=250)

    cadence.widen_display_ttl()

    assert cadence.display_ttl_ms == 250


def test_an_explicit_interval_leaves_the_ttl_untouched() -> None:
    """明示指定された組は、呼び出し側の `display_ttl_ms` をそのまま尊重する。"""
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=125, display_ttl_ms=250)

    cadence.set_interval_ms(333)

    assert cadence.display_ttl_ms == 250


def test_resetting_clears_the_throttle_count_and_admits_at_once() -> None:
    clock = FakeClock()
    cadence = _cadence(clock, interval_ms=1000)
    cadence.admit()
    cadence.admit()
    assert cadence.throttled_frames == 1

    cadence.reset()

    assert cadence.throttled_frames == 0
    assert cadence.admit() is True
