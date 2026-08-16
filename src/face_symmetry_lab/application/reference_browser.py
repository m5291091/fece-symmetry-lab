"""FR-037 / AC-REFERENCE-02〜03: 参照統計の use case と失敗分離。

この層が守るのは 2 つ。**入力検証**（UI から届く絞り込み条件のうち、未知の
ラベルや範囲外の年齢を domain へ流さない）と、**失敗分離**（参照データが
読めなくても例外を UI 全体へ投げない、不変条件 12）である。統計の計算自体は
`domain.reference_stats` にすべて委譲し、ここでは何も計算しない。
"""

from __future__ import annotations

from dataclasses import dataclass

from face_symmetry_lab.application.ports import ReferenceCellRepository
from face_symmetry_lab.domain.reference_stats import (
    AGE_BANDS,
    GENDER_LABELS,
    MAX_AGE,
    MIN_AGE,
    RACE_LABELS,
    ReferenceAggregate,
    ReferenceCell,
    ReferenceDatasetMetadata,
    ReferenceFilter,
    combine,
    select_cells,
)


class ReferenceFilterError(ValueError):
    """絞り込み条件が不正である（未知のラベル、または不正な年齢範囲）。"""


def validate_filter(query: ReferenceFilter) -> None:
    """UI から届く絞り込み条件を検証する。domain の `select_cells` の前段に置く。"""
    unknown_races = query.races - set(RACE_LABELS)
    if unknown_races:
        raise ReferenceFilterError(f"未知の人種です: {sorted(unknown_races)}")

    unknown_genders = query.genders - set(GENDER_LABELS)
    if unknown_genders:
        raise ReferenceFilterError(f"未知の性別です: {sorted(unknown_genders)}")

    unknown_age_bands = query.age_bands - set(AGE_BANDS)
    if unknown_age_bands:
        raise ReferenceFilterError(f"未知の年齢帯です: {sorted(unknown_age_bands)}")

    if query.age_range is not None:
        low, high = query.age_range
        if not (MIN_AGE <= low <= MAX_AGE) or not (MIN_AGE <= high <= MAX_AGE):
            raise ReferenceFilterError(
                f"age_range は {MIN_AGE}〜{MAX_AGE} の範囲である必要があります: {query.age_range!r}"
            )
        if low > high:
            raise ReferenceFilterError(f"age_range の下限が上限を超えています: {query.age_range!r}")


@dataclass(frozen=True, slots=True)
class ReferenceQueryResult:
    """1 回の絞り込みの結果。`aggregate` は一致 0 件のとき `None`。"""

    aggregate: ReferenceAggregate | None
    matched_cells: tuple[ReferenceCell, ...]


class ReferenceBrowser:
    """参照統計セルテーブルへの単一の窓口。

    `__init__` で `repository.load()` を **1 回だけ** 呼ぶ。読み込みが失敗しても
    例外を外へ投げず、理由を文字列として保持するだけにする（不変条件 12）。
    失敗の型を先読みしないのは、壊れたデータが `ReferenceDatasetError` 以外の
    例外（例えば `OSError`）を投げる余地を残すためである。読み込みは再試行しない
    ——再実行のたびに壊れたファイルへ再アクセスして時間を使わないようにする。
    """

    def __init__(self, repository: ReferenceCellRepository) -> None:
        self._cells: tuple[ReferenceCell, ...] = ()
        self._metadata: ReferenceDatasetMetadata | None = None
        self._unavailable_reason: str | None = None
        try:
            dataset = repository.load()
        except Exception as error:  # 失敗分離。型を先読みしない（不変条件 12）。
            self._unavailable_reason = str(error)
            return
        self._metadata = dataset.metadata
        self._cells = dataset.cells

    @property
    def is_available(self) -> bool:
        return self._unavailable_reason is None

    @property
    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason

    @property
    def metadata(self) -> ReferenceDatasetMetadata | None:
        return self._metadata

    def query(self, query: ReferenceFilter) -> ReferenceQueryResult | None:
        """絞り込んで合併する。利用不可なら `None`。

        `combine` へ `min_count` を渡さない——既定のしきい値
        （`MIN_COUNT_FOR_STATISTICS`）をここで緩めると AC-REFERENCE-02 が崩れる。
        """
        if not self.is_available:
            return None
        validate_filter(query)
        matched = select_cells(self._cells, query)
        aggregate = combine(matched)
        return ReferenceQueryResult(aggregate=aggregate, matched_cells=matched)
