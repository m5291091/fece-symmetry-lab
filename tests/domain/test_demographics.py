import math
from datetime import UTC, datetime
from math import nan

import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.demographics import (
    ObservedAges,
    build_demographic_estimate,
    mean_gender_probabilities,
    observed_age,
    observed_ages,
)
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel


def test_build_demographic_estimate_keeps_continuous_age_and_gender_probabilities() -> None:
    estimate = build_demographic_estimate(
        estimated_age_years=36.5,
        gender_probabilities=(0.2, 0.8),
        model_version="openvino-age-gender/1.0.0",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert estimate.estimated_age_years == pytest.approx(36.5)
    assert estimate.gender is GenderLabel.MALE
    assert estimate.gender_confidence == pytest.approx(0.8)
    assert estimate.gender_probabilities == pytest.approx((0.2, 0.8))


@pytest.mark.parametrize("age", [17.99, 75.01, nan])
def test_demographic_estimate_rejects_age_outside_model_scope(age: float) -> None:
    with pytest.raises((ValueError, ValidationError)):
        build_demographic_estimate(
            estimated_age_years=age,
            gender_probabilities=(0.5, 0.5),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )


def test_demographic_estimate_requires_two_finite_probabilities() -> None:
    with pytest.raises((ValueError, ValidationError)):
        build_demographic_estimate(
            estimated_age_years=40,
            gender_probabilities=(1.0,),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1, tzinfo=UTC),
        )


def test_demographic_estimate_requires_timezone_aware_observation() -> None:
    with pytest.raises(ValidationError):
        DemographicEstimate(
            estimated_age_years=40,
            gender=GenderLabel.FEMALE,
            gender_confidence=0.7,
            gender_probabilities=(0.7, 0.3),
            model_version="openvino-age-gender/1.0.0",
            model_sha256="a" * 64,
            observed_at=datetime(2026, 8, 1),
        )


# --- mean_gender_probabilities -----------------------------------------------


def test_mean_gender_probabilities_returns_none_for_empty_input() -> None:
    assert mean_gender_probabilities([]) is None


def test_mean_gender_probabilities_averages_elementwise() -> None:
    result = mean_gender_probabilities([(0.2, 0.8), (0.9, 0.1), (0.2, 0.8)])

    assert result is not None
    assert result == pytest.approx((0.4333, 0.5667), abs=1e-3)


# --- observed_age / ObservedAges / observed_ages ------------------------------
# §9.6.1 の参考年齢要約が使う丸めと範囲決定。FR-039（参照平均の併記）の撤回
# （2026-09-03、§0 項目 34）に伴い `domain.reference_baseline` から移設した。
# FR-039 とは独立に §9.6.1 のレポート要約が使い続けるため、丸めの回帰をここで守る。


@pytest.mark.parametrize(
    ("age_years", "expected"),
    [(20.0, 20), (20.4, 20), (20.6, 21), (18.0, 18), (75.0, 75)],
)
def test_observed_age_rounds_to_the_nearest_year(age_years: float, expected: int) -> None:
    assert observed_age(age_years) == expected


@pytest.mark.parametrize("age_years", [20.5, 21.5, 26.5, 27.5])
def test_observed_age_uses_the_same_rounding_as_the_displayed_age(age_years: float) -> None:
    """画面の「27 歳」とレポートの「28 歳」が食い違ってはならない（§9.6.1）。

    HUD は `f"{age:.0f} 歳"`（偶数丸め）で参考年齢を出す。ここが `int()` や
    `math.floor(x + 0.5)` だと、ちょうど .5 のときだけ 1 歳ずれる。
    """
    resolved = observed_age(age_years)
    assert resolved is not None
    assert f"{resolved}" == f"{age_years:.0f}"


@pytest.mark.parametrize("age_years", [17.4, 17.0, 75.6, 76.0, 0.0, -5.0])
def test_observed_age_rejects_ages_outside_the_model_range(age_years: float) -> None:
    assert observed_age(age_years) is None


@pytest.mark.parametrize("age_years", [math.nan, math.inf, -math.inf])
def test_observed_age_rejects_non_finite_values(age_years: float) -> None:
    """不変条件 11: 非有限値を丸めて 0 や境界値へ落とさない。"""
    assert observed_age(age_years) is None


def test_the_observed_range_is_the_interquartile_one_not_the_extremes() -> None:
    """罠: min〜max を使う実装は、1 フレームの外れ値で範囲を 20〜55 へ広げる。"""
    ages = [*([21.0] * 10), *([23.0] * 10), 55.0]

    observed = observed_ages(ages)

    assert observed is not None
    assert isinstance(observed, ObservedAges)
    assert observed.low >= 21
    assert observed.high <= 24
    assert observed.sample_count == 21


def test_the_observed_range_reports_a_median() -> None:
    """範囲だけでは「どのあたりの顔の値か」が読み取れない。"""
    observed = observed_ages([20.0, 21.0, 22.0, 30.0])

    assert observed is not None
    assert observed.median == 22


def test_a_single_observation_becomes_a_single_year() -> None:
    observed = observed_ages([27.4])

    assert observed is not None
    assert (observed.low, observed.high, observed.median) == (27, 27, 27)


def test_observations_use_the_same_rounding_as_the_screen() -> None:
    """画面の「27 歳」とレポートの年齢が食い違わない（偶数丸め、§9.6.1）。"""
    observed = observed_ages([26.5])

    assert observed is not None
    assert observed.median == 26


def test_ages_outside_the_model_range_are_dropped_not_clamped() -> None:
    """18〜75 の範囲外へ「18 歳」を当てない。"""
    observed = observed_ages([12.0, 20.0, 21.0, 90.0, float("nan")])

    assert observed is not None
    assert observed.sample_count == 2
    assert observed.low >= 20


def test_nothing_observed_yields_no_range() -> None:
    assert observed_ages([]) is None
    assert observed_ages([10.0, 88.0]) is None
