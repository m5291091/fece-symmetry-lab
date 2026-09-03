"""FR-024 / §9.6.1: 表示できた判定だけを溜める（`application.demographic_observations`）。

**この module の存在理由は 1 つの分岐にある。** 年齢ゲートが表示を止めたフレームの
推定を溜めてしまうと、後から FR-042 のレポートの参考年齢要約としてゲートの内部判断が
漏れる（§10.11-2）。ここが壊れると、画面では隠したはずの判断が「20 歳」という形で
ファイルに残る。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from face_symmetry_lab.application.demographic_observations import (
    DemographicObservations,
    summarize_observations,
)
from face_symmetry_lab.domain.demographics import build_demographic_estimate, observed_ages
from face_symmetry_lab.domain.models import GenderLabel


def _estimate(age: float = 24.0, female: float = 0.2):
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(female, 1.0 - female),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 16, tzinfo=UTC),
    )


def test_a_shown_estimate_is_kept() -> None:
    observations = DemographicObservations()

    observations.observe(_estimate(age=24.0), may_show=True)

    snapshot = observations.snapshot()
    assert snapshot.ages == pytest.approx((24.0,))
    assert snapshot.genders == (GenderLabel.MALE,)


def test_an_estimate_the_gate_hid_is_never_kept() -> None:
    """§10.11-2: 隠した判断が、後から母集団の年齢として復元されてはならない。"""
    observations = DemographicObservations()

    # モデル自体は 18〜75 しか返さない（`build_demographic_estimate` の検証）。
    # ゲートが止めるのは「その値を画面へ出してよいか」であって、値の大小ではない。
    observations.observe(_estimate(age=19.0), may_show=False)

    assert observations.snapshot().is_empty


def test_a_missing_estimate_is_ignored() -> None:
    observations = DemographicObservations()

    observations.observe(None, may_show=True)

    assert observations.snapshot().is_empty


def test_only_the_shown_estimates_survive_a_mixed_session() -> None:
    """罠: `may_show` を見ずに全部溜める実装は、ここで隠した 19 歳を混ぜる。"""
    observations = DemographicObservations()

    observations.observe(_estimate(age=24.0), may_show=True)
    observations.observe(_estimate(age=19.0), may_show=False)
    observations.observe(_estimate(age=26.0), may_show=True)

    assert observations.snapshot().ages == pytest.approx((24.0, 26.0))


def test_the_buffer_stays_bounded_and_keeps_the_newest() -> None:
    """長いセッションでも有界に保つ。新しいほうが「いまの顔」に近い。"""
    observations = DemographicObservations(capacity=2)

    for age in (20.0, 21.0, 22.0):
        observations.observe(_estimate(age=age), may_show=True)

    assert observations.snapshot().ages == pytest.approx((21.0, 22.0))


def test_reset_drops_everything() -> None:
    """§4.4: セッションのリセットで観測も捨てる。"""
    observations = DemographicObservations()
    observations.observe(_estimate(), may_show=True)

    observations.reset()

    assert observations.snapshot().is_empty


def test_a_snapshot_does_not_change_under_later_observations() -> None:
    """不変ビュー。読み手（Streamlit 側）が持ち回っても中身が動かない。"""
    observations = DemographicObservations()
    observations.observe(_estimate(age=24.0), may_show=True)

    snapshot = observations.snapshot()
    observations.observe(_estimate(age=30.0), may_show=True)

    assert snapshot.ages == pytest.approx((24.0,))


def test_a_zero_capacity_is_refused() -> None:
    with pytest.raises(ValueError):
        DemographicObservations(capacity=0)


# --- §9.6.1 / FR-042: レポートへ出す要約（2026-08-26） ------------------------


def test_the_overview_reads_the_same_rounding_as_the_population() -> None:
    """§9.6.1: 画面の「24 歳」とレポートの年齢が食い違ってはならない。

    罠: `int()` の切り捨てや自前の丸めを持つ実装は、ちょうど .5 の観測で
    `observed_ages`（レポートの丸め）と 1 歳ずれる。
    """
    observations = DemographicObservations()
    for age in (24.5, 25.5, 26.0):
        observations.observe(_estimate(age=age), may_show=True)

    overview = summarize_observations(observations.snapshot())

    assert overview is not None
    assert overview.ages == observed_ages(observations.snapshot().ages)


def test_the_overview_reports_mean_gender_probability() -> None:
    """判定が途中で入れ替わったセッションは、確率の平均として出す（2026-08-31）。"""
    observations = DemographicObservations()
    observations.observe(_estimate(age=24.0, female=0.2), may_show=True)
    observations.observe(_estimate(age=25.0, female=0.9), may_show=True)
    observations.observe(_estimate(age=26.0, female=0.2), may_show=True)

    overview = summarize_observations(observations.snapshot())

    assert overview is not None
    assert overview.gender_probabilities == pytest.approx((0.4333, 0.5667), abs=1e-3)


def test_a_session_that_showed_nothing_has_no_overview() -> None:
    """1 件も表示できなかったセッションでは、レポートの節ごと出さない（§9.6.1）。"""
    observations = DemographicObservations()
    observations.observe(_estimate(age=24.0), may_show=False)

    assert summarize_observations(observations.snapshot()) is None
