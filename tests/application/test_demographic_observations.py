"""FR-039 / §9.6.1: 表示できた判定だけを溜める（`application.demographic_observations`）。

**この module の存在理由は 1 つの分岐にある。** 年齢ゲートが表示を止めたフレームの
推定を溜めてしまうと、後から FR-042 のレポートの母集団としてゲートの内部判断が
漏れる（§10.12-3、§10.11-2）。ここが壊れると、画面では隠したはずの判断が
「20 歳・男性の平均」という形でファイルに残る。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from face_symmetry_lab.application.demographic_observations import (
    DemographicObservations,
)
from face_symmetry_lab.domain.demographics import build_demographic_estimate
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
    """§10.12-3: 隠した判断が、後から母集団の年齢として復元されてはならない。"""
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
