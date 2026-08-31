"""FR-039 / §10.12: 参照平均の解決（use case・失敗分離・メモ化）。

`domain.reference_baseline` が「どのセルを読むか」を決め、`domain.reference_stats`
が合併する。この層が受け持つのは、**参考年齢・参考性別から母集団を 1 つ決めること**、
**併記できない条件をすべて「併記しない」へ倒すこと**（不変条件 12）、そして
**5 Hz の HUD が同じ絞り込みを毎 tick 繰り返さないこと**だけである。
"""

from __future__ import annotations

import collections
from datetime import UTC, datetime

import pytest

from face_symmetry_lab.adapters.io.reference_dataset import CsvReferenceCellRepository
from face_symmetry_lab.application.reference_baseline import BaselineProvider
from face_symmetry_lab.application.reference_browser import ReferenceBrowser
from face_symmetry_lab.domain.models import DemographicEstimate, GenderLabel
from face_symmetry_lab.domain.reference_stats import (
    METRIC_NAMES,
    MIN_COUNT_FOR_STATISTICS,
    MetricMoments,
    ReferenceCell,
    ReferenceDataset,
    ReferenceDatasetMetadata,
    ReferenceFilter,
)

_METADATA = ReferenceDatasetMetadata(
    schema_version="utkface-symmetry-3d-cells/1.0.0",
    metric_version="symmetry-3d/1.0.0",
    landmark_map_version="mediapipe-478-mirror/1.0.0",
    source_schema_version="utkface-symmetry-3d-labelled/3.0.0",
    source_generated_at="2026-08-14T03:38:57.977344+00:00",
    records_sha256="0" * 64,
    cells_sha256="1" * 64,
    total_count=100,
    min_group_for_ci=8,
    dataset_name="UTKFace",
    dataset_url="https://susanqq.github.io/UTKFace/",
    license_scope="non-commercial research only",
    selection={"age": "18-75"},
    warning="探索値である。",
)


def _cell(race: str, gender: str, age: int, count: int, mean: float) -> ReferenceCell:
    return ReferenceCell(
        race=race,
        gender=gender,
        age=age,
        count=count,
        moments={name: MetricMoments(mean=mean, m2=0.0) for name in METRIC_NAMES},
    )


#: 20 歳の male は 2 人種にまたがって 30 件（10 件の 40.0 と 20 件の 55.0 を件数で
#: 重み付けして 50.0）。同じ 20 歳の female は別の平均を持つので、性別を無視した実装は
#: 72.86 になって落ちる。単純平均（47.5）で合併する実装も落ちる。
_CELLS = (
    _cell("Asian", "male", 20, 10, 40.0),
    _cell("White", "male", 20, 20, 55.0),
    _cell("Asian", "female", 20, 40, 90.0),
    _cell("Asian", "male", 21, 40, 10.0),
    _cell("Asian", "male", 30, 4, 70.0),
)


class _FakeRepository:
    def __init__(self, dataset: ReferenceDataset) -> None:
        self._dataset = dataset
        self.load_calls = 0

    def load(self) -> ReferenceDataset:
        self.load_calls += 1
        return self._dataset


class _FailingRepository:
    def load(self) -> ReferenceDataset:
        raise FileNotFoundError("configs/reference/utkface_symmetry3d_cells_v1.csv がありません")


class _CountingBrowser(ReferenceBrowser):
    """絞り込みが何回走ったかを数える（メモ化の検証用）。"""

    def __init__(self, repository: _FakeRepository) -> None:
        super().__init__(repository)
        self.queries: collections.Counter[tuple[frozenset[str], tuple[int, int] | None]] = (
            collections.Counter()
        )

    def query(self, query: ReferenceFilter):  # type: ignore[no-untyped-def]
        self.queries[(query.genders, query.age_range)] += 1
        return super().query(query)


def _estimate(age: float, gender: GenderLabel) -> DemographicEstimate:
    probabilities = (0.2, 0.8) if gender is GenderLabel.MALE else (0.8, 0.2)
    return DemographicEstimate(
        estimated_age_years=age,
        gender=gender,
        gender_confidence=0.8,
        gender_probabilities=probabilities,
        model_version="age-gender-recognition-retail-0013/1.0.0",
        model_sha256="a" * 64,
        observed_at=datetime.now(UTC),
    )


def _provider(cells: tuple[ReferenceCell, ...] = _CELLS) -> BaselineProvider:
    dataset = ReferenceDataset(metadata=_METADATA, cells=cells)
    return BaselineProvider(ReferenceBrowser(_FakeRepository(dataset)))


# --- 母集団の解決 -------------------------------------------------------------


def test_resolve_merges_every_race_for_the_estimated_age_and_gender() -> None:
    """AC-BASELINE-01: 人種は絞らず、年齢と性別だけで母集団を決める。"""
    summary = _provider().resolve(_estimate(20.4, GenderLabel.MALE))

    assert summary is not None
    assert summary.age == 20
    assert summary.gender is GenderLabel.MALE
    assert summary.count == 30
    for name in METRIC_NAMES:
        assert summary.metrics[name].mean == pytest.approx(50.0)


def test_resolve_does_not_mix_the_other_gender() -> None:
    """罠: 性別を絞らない実装は 20 歳全体の平均（72.86）を返して通ってしまう。"""
    summary = _provider().resolve(_estimate(20.0, GenderLabel.MALE))

    assert summary is not None
    assert summary.count == 30
    assert summary.metrics["score_points"].mean == pytest.approx(50.0)


def test_resolve_does_not_mix_neighbouring_ages() -> None:
    """罠: 年齢帯へ丸める実装は 20 歳と 21 歳を混ぜ、件数が 70 になる。"""
    summary = _provider().resolve(_estimate(20.0, GenderLabel.MALE))

    assert summary is not None
    assert summary.count == 30


# --- 併記しない条件 -----------------------------------------------------------


def test_resolve_returns_nothing_without_an_estimate() -> None:
    """§10.12-3: 参考年齢・参考性別を表示していないときは母集団も出さない。"""
    assert _provider().resolve(None) is None


def test_resolve_returns_nothing_below_the_count_threshold() -> None:
    """AC-BASELINE-03 / §10.12-4: しきい値（8 件）を緩めない。"""
    provider = _provider()
    assert provider.resolve(_estimate(30.0, GenderLabel.MALE)) is None


def test_the_threshold_is_the_frozen_one_from_the_cell_table() -> None:
    """罠: `min_count=1` を渡す実装は、上の 4 件のセルで統計値を出してしまう。"""
    assert MIN_COUNT_FOR_STATISTICS == 8


def test_resolve_returns_nothing_for_an_age_outside_the_cell_table() -> None:
    provider = _provider((_cell("Asian", "male", 18, 40, 50.0),))
    assert provider.resolve(_estimate(18.0, GenderLabel.MALE)) is not None
    assert provider.resolve(_estimate(75.0, GenderLabel.MALE)) is None


def test_resolve_returns_nothing_when_no_cell_matches() -> None:
    provider = _provider()
    assert provider.resolve(_estimate(60.0, GenderLabel.FEMALE)) is None


def test_resolve_survives_an_unreadable_reference_dataset() -> None:
    """AC-BASELINE-03 / 不変条件 12: 例外を上へ投げず、併記だけが消える。"""
    provider = BaselineProvider(ReferenceBrowser(_FailingRepository()))
    assert provider.resolve(_estimate(20.0, GenderLabel.MALE)) is None


# --- メモ化 -------------------------------------------------------------------


def test_repeated_resolution_does_not_refilter_the_cells() -> None:
    """5 Hz の HUD が同じ絞り込みを毎 tick 走らせないこと。"""
    dataset = ReferenceDataset(metadata=_METADATA, cells=_CELLS)
    browser = _CountingBrowser(_FakeRepository(dataset))
    provider = BaselineProvider(browser)

    for _ in range(5):
        provider.resolve(_estimate(20.0, GenderLabel.MALE))

    assert sum(browser.queries.values()) == 1


def test_a_different_group_is_resolved_separately() -> None:
    """罠: 1 件だけ覚える実装は、性別が変わっても前の母集団を返す。"""
    dataset = ReferenceDataset(metadata=_METADATA, cells=_CELLS)
    browser = _CountingBrowser(_FakeRepository(dataset))
    provider = BaselineProvider(browser)

    male = provider.resolve(_estimate(20.0, GenderLabel.MALE))
    female = provider.resolve(_estimate(20.0, GenderLabel.FEMALE))
    again = provider.resolve(_estimate(20.0, GenderLabel.MALE))

    assert male is not None and female is not None
    assert female.count == 40
    assert again == male
    assert sum(browser.queries.values()) == 2


# --- 同梱データそのものの前提 -------------------------------------------------


def test_every_gender_and_age_cell_in_the_bundled_data_clears_the_threshold() -> None:
    """ADR-0006 (b) が 1 歳刻みを選んだ根拠を、同梱データそのもので固定する。

    116 通り（2 性別 × 58 歳）のいずれかが 8 件を割ると、その年齢の利用者にだけ
    併記が黙って消える。データを差し替えたときにここで気付けるようにしておく。
    ネットワークへは行かず、読むのは `configs/reference/` の凍結アセットだけである。
    """
    dataset = CsvReferenceCellRepository().load()
    totals: collections.Counter[tuple[str, int]] = collections.Counter()
    for cell in dataset.cells:
        totals[(cell.gender, cell.age)] += cell.count

    assert len(totals) == 116
    assert min(totals.values()) >= MIN_COUNT_FOR_STATISTICS


# --- FR-039 / §9.6.1: セッション全体の母集団（2026-08-16） --------------------


def _observations(ages, genders):
    from face_symmetry_lab.application.demographic_observations import (
        DemographicObservationSet,
    )

    probs = tuple((0.3, 0.7) for _ in genders)
    return DemographicObservationSet(
        ages=tuple(ages), genders=tuple(genders), gender_probabilities=probs
    )


def test_a_session_resolves_one_population_from_the_observed_range() -> None:
    """§9.6.1: 1 点では決められないので、四分位範囲と性別の集合で引く。"""
    provider = _provider()

    summary = provider.resolve_session(
        _observations([20.0, 20.0, 21.0, 21.0], [GenderLabel.MALE] * 4)
    )

    assert summary is not None
    assert summary.genders == (GenderLabel.MALE,)
    assert summary.ages.low <= summary.ages.median <= summary.ages.high
    assert summary.count > 0
    assert summary.metrics


def test_a_session_with_both_genders_spans_both() -> None:
    """判定が入れ替わったなら、その両方がそのセッションで表示された判定である。"""
    provider = _provider()

    both = provider.resolve_session(
        _observations([20.0] * 4, [GenderLabel.MALE, GenderLabel.FEMALE] * 2)
    )
    male_only = provider.resolve_session(_observations([20.0] * 4, [GenderLabel.MALE] * 4))

    assert both is not None and male_only is not None
    assert set(both.genders) == {GenderLabel.MALE, GenderLabel.FEMALE}
    assert both.count > male_only.count, "両方を含めた母集団のほうが大きい"


def test_a_wider_observed_range_takes_in_more_of_the_table() -> None:
    provider = _provider()

    narrow = provider.resolve_session(_observations([20.0] * 4, [GenderLabel.MALE] * 4))
    wide = provider.resolve_session(_observations([20.0, 20.0, 21.0, 21.0], [GenderLabel.MALE] * 4))

    assert narrow is not None and wide is not None
    assert wide.count > narrow.count


def test_a_session_without_any_shown_estimate_has_no_population() -> None:
    """観測が 1 件も無ければ併記しない（不変条件 12）。"""
    provider = _provider()

    assert provider.resolve_session(_observations([], [])) is None


def test_a_session_outside_the_table_has_no_population() -> None:
    """§9.5 の範囲外へ「18 歳の平均」を当てない。"""
    provider = _provider()

    assert provider.resolve_session(_observations([88.0], [GenderLabel.MALE])) is None
