"""FR-037 / AC-REFERENCE-03: 参照統計の use case と失敗分離。

この層が守るのは 2 つ。**入力検証**（UI が壊れても未知のラベルや範囲外の年齢を
domain へ流さない）と、**失敗分離**（参照データが読めなくても例外を UI 全体へ
投げない、不変条件 12）である。
"""

from __future__ import annotations

import pytest

from face_symmetry_lab.application.reference_browser import (
    ReferenceBrowser,
    ReferenceFilterError,
    validate_filter,
)
from face_symmetry_lab.domain.reference_stats import (
    METRIC_NAMES,
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
    total_count=24,
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
        moments={name: MetricMoments(mean=mean, m2=float(count)) for name in METRIC_NAMES},
    )


class _FakeRepository:
    """`load()` の呼ばれた回数を数える fake。"""

    def __init__(self, dataset: ReferenceDataset) -> None:
        self._dataset = dataset
        self.load_calls = 0

    def load(self) -> ReferenceDataset:
        self.load_calls += 1
        return self._dataset


class _FailingRepository:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.load_calls = 0

    def load(self) -> ReferenceDataset:
        self.load_calls += 1
        raise self._error


def _dataset() -> ReferenceDataset:
    return ReferenceDataset(
        metadata=_METADATA,
        cells=(
            _cell("Asian", "male", 30, 10, 40.0),
            _cell("Asian", "female", 30, 10, 50.0),
            _cell("White", "male", 40, 4, 60.0),
        ),
    )


def test_browser_loads_the_dataset_exactly_once() -> None:
    """再実行のたびに読み直さない（FR-037）。"""
    repository = _FakeRepository(_dataset())
    browser = ReferenceBrowser(repository)

    browser.query(ReferenceFilter())
    browser.query(ReferenceFilter(genders=frozenset({"male"})))

    assert repository.load_calls == 1
    assert browser.is_available


def test_query_returns_matching_cells_and_their_aggregate() -> None:
    browser = ReferenceBrowser(_FakeRepository(_dataset()))

    result = browser.query(ReferenceFilter(races=frozenset({"Asian"})))

    assert result is not None
    assert len(result.matched_cells) == 2
    assert result.aggregate is not None
    assert result.aggregate.count == 20
    assert result.aggregate.metrics["score_points"].mean == pytest.approx(45.0)


def test_query_uses_the_default_threshold() -> None:
    """AC-REFERENCE-02: application はしきい値を緩めない。

    4 件しかないセルを選ぶと、件数は出るが統計値は出ない。`min_count=1` を
    渡してしまう実装はここで落ちる。
    """
    browser = ReferenceBrowser(_FakeRepository(_dataset()))

    result = browser.query(ReferenceFilter(races=frozenset({"White"})))

    assert result is not None
    assert result.aggregate is not None
    assert result.aggregate.count == 4
    assert not result.aggregate.has_statistics


def test_query_reports_no_match_as_an_empty_result() -> None:
    browser = ReferenceBrowser(_FakeRepository(_dataset()))

    result = browser.query(ReferenceFilter(races=frozenset({"Others"})))

    assert result is not None
    assert result.matched_cells == ()
    assert result.aggregate is None


@pytest.mark.parametrize(
    "bad",
    [
        ReferenceFilter(races=frozenset({"Martian"})),
        ReferenceFilter(races=frozenset({"Asian", "Martian"})),
        ReferenceFilter(genders=frozenset({"unspecified"})),
        ReferenceFilter(age_bands=frozenset({"76-99"})),
    ],
)
def test_validate_filter_rejects_unknown_labels(bad: ReferenceFilter) -> None:
    """テスト 9: 未知の人種・性別・年齢帯ラベルを拒否する。"""
    with pytest.raises(ReferenceFilterError):
        validate_filter(bad)


@pytest.mark.parametrize("age_range", [(17, 40), (40, 76), (0, 200), (-5, 20)])
def test_validate_filter_rejects_ages_outside_the_published_range(
    age_range: tuple[int, int],
) -> None:
    """テスト 10a: 18〜75 の外を拒否する。"""
    with pytest.raises(ReferenceFilterError):
        validate_filter(ReferenceFilter(age_range=age_range))


def test_validate_filter_rejects_inverted_age_range() -> None:
    """テスト 10b: 下限 > 上限を拒否する。"""
    with pytest.raises(ReferenceFilterError):
        validate_filter(ReferenceFilter(age_range=(50, 30)))


def test_validate_filter_accepts_the_boundaries_and_a_single_year() -> None:
    validate_filter(ReferenceFilter(age_range=(18, 75)))
    validate_filter(ReferenceFilter(age_range=(30, 30)))
    validate_filter(ReferenceFilter())


def test_query_rejects_an_invalid_filter() -> None:
    browser = ReferenceBrowser(_FakeRepository(_dataset()))

    with pytest.raises(ReferenceFilterError):
        browser.query(ReferenceFilter(age_range=(10, 90)))


def test_browser_absorbs_a_load_failure_instead_of_raising() -> None:
    """テスト 11 / AC-REFERENCE-03: 読み込み失敗を例外として外へ出さない。

    参照ページが使えないだけで、カメラ・画像の左右差分析は止まらない
    （不変条件 12）。
    """
    repository = _FailingRepository(FileNotFoundError("configs/reference が無い"))
    browser = ReferenceBrowser(repository)

    assert not browser.is_available
    assert browser.metadata is None
    assert browser.unavailable_reason is not None
    assert browser.query(ReferenceFilter()) is None


def test_browser_absorbs_an_unexpected_failure_type() -> None:
    """壊れたデータが投げる例外の型を先読みしない。"""
    browser = ReferenceBrowser(_FailingRepository(ValueError("schema_version が違う")))

    assert not browser.is_available
    assert browser.unavailable_reason is not None


def test_browser_does_not_retry_a_failed_load() -> None:
    repository = _FailingRepository(OSError("読めない"))
    browser = ReferenceBrowser(repository)

    browser.query(ReferenceFilter())
    browser.query(ReferenceFilter())

    assert repository.load_calls == 1


def test_metadata_is_exposed_for_version_display() -> None:
    """FR-037 / AC-REFERENCE-04: 版と sha256 を画面へ出せること。"""
    browser = ReferenceBrowser(_FakeRepository(_dataset()))

    metadata = browser.metadata

    assert metadata is not None
    assert metadata.metric_version == "symmetry-3d/1.0.0"
    assert metadata.cells_sha256 == "1" * 64
    assert metadata.license_scope == "non-commercial research only"
