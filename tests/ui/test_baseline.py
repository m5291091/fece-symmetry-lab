"""FR-039: 画面側から参照平均を解決する薄い層（`ui.baseline`）。

判定条件はここに無い（それは `application.reference_baseline` と、ゲートを読む
`ui.hud` / `ui.image_section` の仕事）。ここで確かめるのは 3 点だけである。

1. 推定が無いときは provider へ触れない（§10.12-3）。
2. 同梱の凍結アセットを実際に読んで、実データで母集団が解決できる。
3. 参照データを 1 度しか読み込まない。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from face_symmetry_lab.application.reference_baseline import BaselineProvider, BaselineSummary
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel
from face_symmetry_lab.ui.baseline import default_baseline_provider, resolve_baseline


class _SpyProvider:
    def __init__(self, summary: BaselineSummary | None = None) -> None:
        self.calls: list[DemographicEstimate | None] = []
        self._summary = summary

    def resolve(self, estimate: DemographicEstimate | None) -> BaselineSummary | None:
        self.calls.append(estimate)
        return self._summary


def _estimate(age: float = 20.0, gender: GenderLabel = GenderLabel.MALE) -> DemographicEstimate:
    probabilities = (0.2, 0.8) if gender is GenderLabel.MALE else (0.8, 0.2)
    return DemographicEstimate(
        estimated_age_years=age,
        gender=gender,
        gender_confidence=0.8,
        gender_probabilities=probabilities,
        model_version="age-gender-recognition-retail-0013/1.0.0",
        model_sha256="b" * 64,
        observed_at=datetime.now(UTC),
    )


def test_no_estimate_resolves_to_no_baseline() -> None:
    """§10.12-3: 参考年齢・参考性別を表示していないときは母集団を出さない。

    **provider へ問い合わせもしない。** 参照データ（537 行の CSV と sha256 検証）の
    読み込みを、実際に併記が要るときまで遅らせるためである。映像上の HUD はカメラ
    開始直後——まだ推定が無い状態——から 5 Hz で描き直される。
    """
    provider = _SpyProvider()
    assert resolve_baseline(None, provider=provider) is None  # type: ignore[arg-type]
    assert provider.calls == []


def test_an_explicit_provider_is_used_instead_of_the_shared_one() -> None:
    summary = BaselineSummary(age=20, gender=GenderLabel.MALE, count=75, metrics={})
    provider = _SpyProvider(summary)

    resolved = resolve_baseline(_estimate(), provider=provider)  # type: ignore[arg-type]

    assert resolved is summary
    assert len(provider.calls) == 1


def test_the_shared_provider_is_built_once() -> None:
    assert isinstance(default_baseline_provider(), BaselineProvider)
    assert default_baseline_provider() is default_baseline_provider()


def test_the_bundled_reference_data_resolves_a_real_population() -> None:
    """AC-BASELINE-01: 実アセットを通した端から端までの解決。

    20 歳・male は `configs/reference/` の全人種を合わせて 75 件である。ここが
    落ちるときは、参照データの差し替えか、絞り込みが人種・年齢・性別のどれかを
    取りこぼしている。
    """
    summary = resolve_baseline(_estimate(20.4, GenderLabel.MALE))

    assert summary is not None
    assert (summary.age, summary.gender, summary.count) == (20, GenderLabel.MALE, 75)
    assert summary.metrics["score_points"].mean == pytest.approx(45.473, abs=0.01)
    assert summary.metrics["symmetry_percent_overall"].mean == pytest.approx(96.587, abs=0.01)


def test_the_bundled_data_separates_the_two_genders() -> None:
    """罠: 性別を絞らない実装は、どちらの推定でも同じ件数を返す。"""
    male = resolve_baseline(_estimate(20.0, GenderLabel.MALE))
    female = resolve_baseline(_estimate(20.0, GenderLabel.FEMALE))

    assert male is not None and female is not None
    assert male.count == 75
    assert female.count == 130
