"""FR-039 / §10.12: 参照平均の併記に必要な純粋計算。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import pytest

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import GenderLabel
from face_symmetry_lab.domain.reference_baseline import (
    OVERALL_METRIC,
    PART_METRICS,
    SCORE_METRIC,
    BaselineReading,
    baseline_age,
    baseline_filter,
    baseline_reading,
    observed_ages,
    session_baseline_filter,
)
from face_symmetry_lab.domain.reference_stats import (
    GENDER_LABELS,
    METRIC_NAMES,
    MetricSummary,
)


def _metrics(**means: float) -> Mapping[str, MetricSummary]:
    return MappingProxyType(
        {name: MetricSummary(mean=value, sd=1.0) for name, value in means.items()}
    )


# --- 指標名の対応 -------------------------------------------------------------


def test_metric_names_match_the_frozen_cell_table() -> None:
    """§9.5 の列名と 1 文字でも違えば、平均は黙って欠測になる。"""
    assert SCORE_METRIC in METRIC_NAMES
    assert OVERALL_METRIC in METRIC_NAMES
    assert tuple(PART_METRICS) == PART_NAMES
    for column in PART_METRICS.values():
        assert column in METRIC_NAMES


def test_gender_labels_agree_with_the_cell_table() -> None:
    """`GenderLabel` の値が `GENDER_LABELS` とずれると、絞り込みが 0 件になる。

    0 件は「参照データが無い」と区別できないまま併記が消えるだけなので、
    画面上のどこにもエラーが出ない。ここで固定する。
    """
    assert {label.value for label in GenderLabel} == set(GENDER_LABELS)


# --- 母集団の年齢 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("age_years", "expected"),
    [(20.0, 20), (20.4, 20), (20.6, 21), (18.0, 18), (75.0, 75)],
)
def test_baseline_age_rounds_to_the_nearest_year(age_years: float, expected: int) -> None:
    assert baseline_age(age_years) == expected


@pytest.mark.parametrize("age_years", [20.5, 21.5, 26.5, 27.5])
def test_baseline_age_uses_the_same_rounding_as_the_displayed_age(age_years: float) -> None:
    """§10.12-2: 画面の「27 歳」と母集団の「28 歳」が食い違ってはならない。

    HUD は `f"{age:.0f} 歳"`（偶数丸め）で参考年齢を出す。ここが `int()` や
    `math.floor(x + 0.5)` だと、ちょうど .5 のときだけ 1 歳ずれる。
    """
    resolved = baseline_age(age_years)
    assert resolved is not None
    assert f"{resolved}" == f"{age_years:.0f}"


@pytest.mark.parametrize("age_years", [17.4, 17.0, 75.6, 76.0, 0.0, -5.0])
def test_baseline_age_rejects_ages_outside_the_cell_table(age_years: float) -> None:
    assert baseline_age(age_years) is None


@pytest.mark.parametrize("age_years", [math.nan, math.inf, -math.inf])
def test_baseline_age_rejects_non_finite_values(age_years: float) -> None:
    """不変条件 11: 非有限値を丸めて 0 や境界値へ落とさない。"""
    assert baseline_age(age_years) is None


# --- 絞り込み条件 -------------------------------------------------------------


def test_baseline_filter_selects_one_gender_and_one_age() -> None:
    query = baseline_filter(age=20, gender=GenderLabel.MALE)
    assert query.genders == frozenset({"male"})
    assert query.age_range == (20, 20)
    assert query.age_bands == frozenset()


def test_baseline_filter_never_narrows_by_race() -> None:
    """§10.12-2 / 不変条件 7: アプリは人種を推定しないので、絞る手段を持たない。"""
    for gender in GenderLabel:
        assert baseline_filter(age=30, gender=gender).races == frozenset()


# --- 平均と差 -----------------------------------------------------------------


def test_baseline_reading_returns_mean_and_signed_difference() -> None:
    metrics = _metrics(**{OVERALL_METRIC: 96.31})
    reading = baseline_reading(metrics, OVERALL_METRIC, value=98.42)
    assert reading == BaselineReading(mean=pytest.approx(96.31), delta=pytest.approx(2.11))


def test_baseline_reading_difference_is_display_value_minus_mean() -> None:
    """符号の向きを固定する（罠: 引き算を逆に書いた実装は符号だけ反転して通る）。"""
    metrics = _metrics(**{SCORE_METRIC: 41.5})
    reading = baseline_reading(metrics, SCORE_METRIC, value=20.7)
    assert reading is not None
    assert reading.delta == pytest.approx(-20.8)


def test_baseline_reading_keeps_the_mean_when_the_display_value_is_missing() -> None:
    """不変条件 11: 表示値が `—` の項目に差を出さない。0 で埋めない。"""
    metrics = _metrics(**{OVERALL_METRIC: 96.31})
    reading = baseline_reading(metrics, OVERALL_METRIC, value=None)
    assert reading is not None
    assert reading.mean == pytest.approx(96.31)
    assert reading.delta is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_baseline_reading_rejects_non_finite_display_values(value: float) -> None:
    reading = baseline_reading(_metrics(**{OVERALL_METRIC: 96.31}), OVERALL_METRIC, value=value)
    assert reading is not None
    assert reading.delta is None


def test_baseline_reading_is_missing_when_the_metric_is_absent() -> None:
    """しきい値未満の集計は `metrics` が空になる（§9.5、AC-REFERENCE-02）。"""
    assert baseline_reading(MappingProxyType({}), OVERALL_METRIC, value=98.42) is None


# --- FR-039 / §9.6.1: セッション全体の母集団（2026-08-16） --------------------


def test_the_observed_range_is_the_interquartile_one_not_the_extremes() -> None:
    """罠: min〜max を使う実装は、1 フレームの外れ値で母集団を 20〜55 へ広げる。"""
    ages = [*([21.0] * 10), *([23.0] * 10), 55.0]

    observed = observed_ages(ages)

    assert observed is not None
    assert observed.low >= 21
    assert observed.high <= 24
    assert observed.sample_count == 21


def test_the_observed_range_reports_a_median_for_the_population_label() -> None:
    """範囲だけでは「どのあたりの顔の平均か」が読み取れない。"""
    observed = observed_ages([20.0, 21.0, 22.0, 30.0])

    assert observed is not None
    assert observed.median == 22


def test_a_single_observation_becomes_a_single_year() -> None:
    observed = observed_ages([27.4])

    assert observed is not None
    assert (observed.low, observed.high, observed.median) == (27, 27, 27)


def test_observations_use_the_same_rounding_as_the_screen() -> None:
    """§10.12-2: 画面の「27 歳」と母集団の年齢が食い違わない（偶数丸め）。"""
    observed = observed_ages([26.5])

    assert observed is not None
    assert observed.median == 26


def test_ages_outside_the_cell_table_are_dropped_not_clamped() -> None:
    """§9.5 の範囲外へ「18 歳の平均」を当てない。"""
    observed = observed_ages([12.0, 20.0, 21.0, 90.0, float("nan")])

    assert observed is not None
    assert observed.sample_count == 2
    assert observed.low >= 20


def test_nothing_observed_yields_no_population() -> None:
    assert observed_ages([]) is None
    assert observed_ages([10.0, 88.0]) is None


def test_the_session_filter_spans_the_range_and_every_observed_gender() -> None:
    """判定が入れ替わったなら、その両方がそのセッションで表示された判定である。"""
    observed = observed_ages([20.0, 21.0, 22.0, 23.0])
    assert observed is not None

    query = session_baseline_filter(ages=observed, genders=(GenderLabel.MALE, GenderLabel.FEMALE))

    assert query.age_range == (observed.low, observed.high)
    assert query.genders == frozenset({"male", "female"})
    assert query.races == frozenset(), "不変条件 7: 人種で絞る手段を持たない"
