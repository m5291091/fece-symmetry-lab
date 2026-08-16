import math

import pytest

from face_symmetry_lab.domain.smoothing import (
    DEFAULT_TAU_SECONDS,
    MAX_TAU_SECONDS,
    MIN_TAU_SECONDS,
    TAU_CHOICES_SECONDS,
    SmoothingError,
    TimeBasedEma,
    effective_sample_count,
    nearest_tau_choice,
    response_time_seconds,
    validate_tau_seconds,
)


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


# --- FR-040: 利用者が選べる時定数 ---------------------------------------------


def test_the_selectable_steps_stay_inside_the_documented_range() -> None:
    """§15.2 の値域（0.05〜5.0）の内側であること。段だけを別に決めない。"""
    assert TAU_CHOICES_SECONDS == (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
    assert all(MIN_TAU_SECONDS <= tau <= MAX_TAU_SECONDS for tau in TAU_CHOICES_SECONDS)
    assert list(TAU_CHOICES_SECONDS) == sorted(TAU_CHOICES_SECONDS)


def test_the_default_is_one_of_the_selectable_steps() -> None:
    """既定が段の中に無いと、起動直後の選択が「どれでもない」状態になる。"""
    assert DEFAULT_TAU_SECONDS in TAU_CHOICES_SECONDS


def test_validate_returns_a_value_inside_the_range() -> None:
    assert validate_tau_seconds(0.4) == pytest.approx(0.4)
    assert validate_tau_seconds(MIN_TAU_SECONDS) == pytest.approx(MIN_TAU_SECONDS)
    assert validate_tau_seconds(MAX_TAU_SECONDS) == pytest.approx(MAX_TAU_SECONDS)


@pytest.mark.parametrize("bad", [0.0, -0.4, 0.049, 5.001, math.nan, math.inf, -math.inf])
def test_validate_refuses_values_outside_the_range(bad: float) -> None:
    """罠: clamp する実装は、範囲外の指定を黙って別の設定へ読み替える。"""
    with pytest.raises(SmoothingError):
        validate_tau_seconds(bad)


def test_the_nearest_step_is_used_for_a_time_constant_off_the_ladder() -> None:
    """段の外の値でも選択 UI が落ちないようにする橋渡し。"""
    assert nearest_tau_choice(0.45) == pytest.approx(0.4)
    assert nearest_tau_choice(0.61) == pytest.approx(0.7)
    assert nearest_tau_choice(0.05) == pytest.approx(0.1)
    assert nearest_tau_choice(5.0) == pytest.approx(5.0)


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_a_missing_time_constant_falls_back_to_the_default_step(value: float | None) -> None:
    assert nearest_tau_choice(value) == pytest.approx(DEFAULT_TAU_SECONDS)


def test_the_63_percent_response_time_is_tau_itself() -> None:
    """EMA の定義そのもの。ここが tau と一致しない実装は式を書き換えている。"""
    assert response_time_seconds(0.4, fraction=1.0 - math.exp(-1.0)) == pytest.approx(0.4)


def test_the_95_percent_response_time_is_tau_times_ln20() -> None:
    """罠: `3 * tau` と近似した実装は 1.2 を返し、ここで落ちる（正しくは 1.198…）。"""
    assert response_time_seconds(0.4, fraction=0.95) == pytest.approx(0.4 * math.log(20.0))
    assert response_time_seconds(0.4, fraction=0.95) == pytest.approx(1.1982, abs=1e-4)


def test_a_longer_time_constant_takes_longer_to_catch_up() -> None:
    assert response_time_seconds(2.0) > response_time_seconds(0.4)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_response_time_refuses_fractions_outside_the_open_unit_interval(fraction: float) -> None:
    """0 と 1 は開区間の外。1 は無限大、0 は「追従しない」で意味を成さない。"""
    with pytest.raises(SmoothingError):
        response_time_seconds(0.4, fraction=fraction)


def test_response_time_refuses_a_time_constant_outside_the_range() -> None:
    with pytest.raises(SmoothingError):
        response_time_seconds(0.0)


def test_the_effective_sample_count_is_the_time_constant_over_the_interval() -> None:
    """125ms 周期で tau=0.4 秒なら、およそ 3.2 サンプル分を均している。"""
    assert effective_sample_count(0.4, interval_ms=125) == pytest.approx(3.2)
    assert effective_sample_count(1.0, interval_ms=200) == pytest.approx(5.0)


def test_a_slower_analysis_cadence_means_fewer_samples_in_the_same_window() -> None:
    """罠: 周期を無視して tau だけから本数を出す実装は、両者を同じ値にする。"""
    assert effective_sample_count(1.0, interval_ms=333) < effective_sample_count(
        1.0, interval_ms=125
    )


@pytest.mark.parametrize("interval_ms", [0, -125])
def test_the_effective_sample_count_refuses_a_non_positive_interval(interval_ms: int) -> None:
    with pytest.raises(SmoothingError):
        effective_sample_count(0.4, interval_ms=interval_ms)
