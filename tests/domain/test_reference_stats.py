"""§9.5 / FR-037: 参照統計セルの絞り込みと合併。

この module の要点は「セル集計だけを持ち歩いて、個票と同じ平均・標本標準偏差を
出せること」である。したがって test の主役は、**合成した個票から直接計算した値**
との突き合わせになる。個票を持たない実装が正しいことを、個票を持つ test で
証明する形になっている。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import pytest

from face_symmetry_lab.domain.reference_stats import (
    AGE_BANDS,
    METRIC_NAMES,
    MIN_COUNT_FOR_STATISTICS,
    MetricMoments,
    ReferenceCell,
    ReferenceFilter,
    ReferenceStatsError,
    combine,
    select_cells,
)

_PRIMARY = "score_points"


def _moments(values: Sequence[float]) -> MetricMoments:
    mean = statistics.fmean(values)
    return MetricMoments(mean=mean, m2=sum((value - mean) ** 2 for value in values))


def _cell(
    race: str,
    gender: str,
    age: int,
    values: Sequence[float],
    *,
    overrides: Mapping[str, Sequence[float]] | None = None,
) -> ReferenceCell:
    """個票の値そのものからセルを組み立てるテスト用 helper。

    7 指標すべてを埋める。`overrides` を渡さない指標には同じ値を入れる——
    どの指標を読んでいるかを問わない test では、それで十分区別がつく。
    """
    per_metric = dict(overrides or {})
    return ReferenceCell(
        race=race,
        gender=gender,
        age=age,
        count=len(values),
        moments={name: _moments(per_metric.get(name, values)) for name in METRIC_NAMES},
    )


def test_age_bands_match_the_source_notebook() -> None:
    """§9.5: 年齢帯は元 notebook の `age_band` と完全一致していなければならない。

    ここがずれると、参照ページの帯と配布データの帯が別物になり、件数だけが
    合っているのに中身が違うという最も気づきにくい壊れ方をする。
    """
    assert dict(AGE_BANDS) == {
        "18-24": (18, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65-75": (65, 75),
    }
    assert list(AGE_BANDS) == sorted(AGE_BANDS), "帯は昇順で並んでいること"


def test_min_count_matches_notebook_min_group_for_ci() -> None:
    """§9.5: しきい値 8 は notebook の `min_group_for_ci` と揃えた値である。"""
    assert MIN_COUNT_FOR_STATISTICS == 8


def test_combine_single_cell_returns_its_own_mean_and_sd() -> None:
    """テスト 1: 単一セルの合併は、そのセルの平均と標本標準偏差をそのまま返す。"""
    values = [10.0, 12.0, 14.0, 11.0, 13.0, 15.0, 9.0, 16.0, 12.0, 10.0]
    aggregate = combine([_cell("Asian", "male", 30, values)])

    assert aggregate is not None
    assert aggregate.count == len(values)
    summary = aggregate.metrics[_PRIMARY]
    assert summary.mean == pytest.approx(statistics.fmean(values))
    assert summary.sd is not None
    assert summary.sd == pytest.approx(statistics.stdev(values))


def test_combine_matches_statistics_computed_from_the_raw_records() -> None:
    """テスト 2（本命）: 複数セルの合併が、個票から直接計算した値と一致する。

    セル集計方式が成立する根拠そのもの。`Σx²/n − mean²` 型の実装は、ここで
    桁落ちして落ちるか、少なくとも `pytest.approx` の外へ出る。
    """
    groups = {
        ("Asian", "male", 30): [10.0, 12.0, 14.0],
        ("Asian", "female", 31): [20.5, 21.5],
        ("White", "male", 30): [5.0, 7.0, 9.0, 11.0],
        ("White", "female", 44): [30.25, 31.75, 29.0],
    }
    cells = [_cell(race, gender, age, values) for (race, gender, age), values in groups.items()]
    raw = [value for values in groups.values() for value in values]

    aggregate = combine(cells)

    assert aggregate is not None
    assert aggregate.count == len(raw)
    summary = aggregate.metrics[_PRIMARY]
    assert summary.mean == pytest.approx(statistics.fmean(raw))
    assert summary.sd is not None
    assert summary.sd == pytest.approx(statistics.stdev(raw))


def test_combine_is_stable_with_large_offsets() -> None:
    """テスト 2 の桁落ち版: 平均が大きく分散が小さい入力でも標準偏差が壊れない。

    `sum_of_squares` を持たせて `Σx²/n − mean²` で分散を出す実装は、ここで
    負の分散（= `sqrt` で NaN、あるいは 0 への丸め）を返す。

    合計を 8 件（= `MIN_COUNT_FOR_STATISTICS`）にしてあるのは、既定のしきい値の
    まま統計値を受け取るためである。ここを 8 未満にすると、桁落ちではなく
    「件数不足」を見ることになり、test が意図した罠を仕掛けられなくなる。
    """
    offset = 1.0e8
    groups = {
        ("Asian", "male", 30): [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0],
        ("White", "male", 30): [offset + 5.0, offset + 6.0, offset + 7.0, offset + 8.0],
    }
    cells = [_cell(race, gender, age, values) for (race, gender, age), values in groups.items()]
    raw = [value for values in groups.values() for value in values]

    aggregate = combine(cells)

    assert aggregate is not None
    summary = aggregate.metrics[_PRIMARY]
    assert summary.sd is not None
    assert math.isfinite(summary.sd)
    assert summary.sd == pytest.approx(statistics.stdev(raw), rel=1e-9)


def test_combine_is_order_independent() -> None:
    """テスト 3: 合併は結合律を満たし、順序を変えても同じ結果になる。"""
    cells = [
        _cell("Asian", "male", 30, [1.0, 2.0, 3.0]),
        _cell("Black", "female", 40, [10.0, 20.0]),
        _cell("White", "male", 50, [7.5, 8.5, 9.5, 10.5]),
    ]

    forward = combine(cells)
    backward = combine(list(reversed(cells)))
    shuffled = combine([cells[1], cells[2], cells[0]])

    assert forward is not None and backward is not None and shuffled is not None
    for other in (backward, shuffled):
        assert other.count == forward.count
        assert other.metrics[_PRIMARY].mean == pytest.approx(forward.metrics[_PRIMARY].mean)
        expected_sd = forward.metrics[_PRIMARY].sd
        actual_sd = other.metrics[_PRIMARY].sd
        assert expected_sd is not None and actual_sd is not None
        assert actual_sd == pytest.approx(expected_sd)


def test_combine_of_two_cells_lands_between_them() -> None:
    """テスト 8（罠）: 「常に 0」「常に全体平均」を返す実装では通らない。

    値の異なる 2 セルを、**件数の違う**まま合併する。件数を無視して単純平均
    する実装もここで落ちる（期待値は 20.0 ではなく 15.0 側へ寄る）。
    """
    low = _cell("Asian", "male", 30, [10.0] * 9)
    high = _cell("White", "male", 30, [30.0] * 3)

    aggregate = combine([low, high])

    assert aggregate is not None
    assert aggregate.count == 12
    mean = aggregate.metrics[_PRIMARY].mean
    assert mean == pytest.approx(15.0)
    assert mean != pytest.approx(0.0)
    assert mean != pytest.approx(20.0), "件数で重み付けしない実装は 20.0 を返す"


def test_combine_returns_none_when_no_cells_match() -> None:
    """テスト 5: 一致セルが 0 件なら `None`。0 件を件数 0 の統計として返さない。"""
    assert combine([]) is None


def test_combine_withholds_statistics_below_the_threshold() -> None:
    """テスト 6: 合計件数がしきい値未満なら統計値を返さず、件数だけを返す。

    件数そのものは UI が最初に見せるので、`None` ではなく「統計の無い集計」
    として返る。
    """
    aggregate = combine([_cell("Others", "female", 60, [1.0] * (MIN_COUNT_FOR_STATISTICS - 1))])

    assert aggregate is not None
    assert aggregate.count == MIN_COUNT_FOR_STATISTICS - 1
    assert not aggregate.has_statistics
    assert aggregate.metrics == {}


def test_combine_publishes_statistics_at_the_threshold() -> None:
    aggregate = combine(
        [_cell("Others", "female", 60, [1.0, 2.0] * (MIN_COUNT_FOR_STATISTICS // 2))]
    )

    assert aggregate is not None
    assert aggregate.count == MIN_COUNT_FOR_STATISTICS
    assert aggregate.has_statistics
    assert set(aggregate.metrics) == set(METRIC_NAMES)


def test_combine_reports_missing_sd_for_a_single_record() -> None:
    """標本標準偏差は `n < 2` で欠測。0.0 で埋めない（不変条件 11）。

    しきい値 8 を通す経路では起きないので、しきい値を下げて素の統計を見る。
    `min_count` はそのために露出しており、`application` と `ui` は既定値だけを使う。
    """
    aggregate = combine([_cell("Indian", "male", 25, [42.0])], min_count=1)

    assert aggregate is not None
    assert aggregate.count == 1
    assert aggregate.metrics[_PRIMARY].mean == pytest.approx(42.0)
    assert aggregate.metrics[_PRIMARY].sd is None


def _population() -> list[ReferenceCell]:
    return [
        _cell("Asian", "male", 20, [1.0]),
        _cell("Asian", "female", 30, [2.0, 2.0]),
        _cell("Indian", "male", 30, [3.0, 3.0, 3.0]),
        _cell("Indian", "female", 45, [4.0] * 4),
        _cell("White", "male", 70, [5.0] * 5),
    ]


def test_select_cells_without_any_axis_returns_everything() -> None:
    """テスト 4a: 全軸未選択は「絞り込まない」であり、「何も選ばれていない」ではない。"""
    cells = _population()

    assert select_cells(cells, ReferenceFilter()) == tuple(cells)


def test_select_cells_by_single_axis() -> None:
    """テスト 4b: 1 軸だけの絞り込み。"""
    cells = _population()

    selected = select_cells(cells, ReferenceFilter(genders=frozenset({"female"})))

    assert [cell.age for cell in selected] == [30, 45]


def test_select_cells_by_multiple_values_on_one_axis() -> None:
    """テスト 4c: 同じ軸の複数選択は和集合になる（race = {Asian, Indian}）。"""
    cells = _population()

    selected = select_cells(cells, ReferenceFilter(races=frozenset({"Asian", "Indian"})))

    assert {cell.race for cell in selected} == {"Asian", "Indian"}
    assert len(selected) == 4


def test_select_cells_across_axes_is_a_conjunction() -> None:
    """テスト 4d: 軸をまたぐと積集合になる。"""
    cells = _population()

    selected = select_cells(
        cells,
        ReferenceFilter(
            races=frozenset({"Asian", "Indian"}),
            genders=frozenset({"male"}),
            age_bands=frozenset({"25-34"}),
        ),
    )

    assert len(selected) == 1
    assert (selected[0].race, selected[0].gender, selected[0].age) == ("Indian", "male", 30)


def test_select_cells_by_age_band_union() -> None:
    cells = _population()

    selected = select_cells(cells, ReferenceFilter(age_bands=frozenset({"18-24", "65-75"})))

    assert [cell.age for cell in selected] == [20, 70]


def test_select_cells_by_explicit_age_range_is_inclusive() -> None:
    cells = _population()

    selected = select_cells(cells, ReferenceFilter(age_range=(30, 45)))

    assert [cell.age for cell in selected] == [30, 30, 45]


def test_age_band_and_age_range_apply_together() -> None:
    """両方指定されたら両方を満たすセルだけが残る（UI は片方しか設定しない）。"""
    cells = _population()

    selected = select_cells(
        cells,
        ReferenceFilter(age_bands=frozenset({"25-34"}), age_range=(30, 45)),
    )

    assert [cell.age for cell in selected] == [30, 30]

    empty = select_cells(
        cells,
        ReferenceFilter(age_bands=frozenset({"18-24"}), age_range=(30, 45)),
    )

    assert empty == ()


def test_filtered_mean_differs_from_the_population_mean() -> None:
    """テスト 8（罠の続き）: 絞り込みを無視して全体平均を返す実装は落ちる。"""
    cells = _population()

    everything = combine(cells, min_count=1)
    indians = combine(
        select_cells(cells, ReferenceFilter(races=frozenset({"Indian"}))), min_count=1
    )

    assert everything is not None and indians is not None
    assert indians.count == 7
    assert indians.metrics[_PRIMARY].mean == pytest.approx((3.0 * 3 + 4.0 * 4) / 7)
    assert indians.metrics[_PRIMARY].mean != pytest.approx(everything.metrics[_PRIMARY].mean)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("race", "Martian"),
        ("gender", "unspecified"),
    ],
)
def test_cell_rejects_unknown_labels(field: str, value: str) -> None:
    kwargs = {"race": "Asian", "gender": "male", "age": 30}
    kwargs[field] = value

    with pytest.raises(ReferenceStatsError):
        ReferenceCell(
            count=3,
            moments={name: MetricMoments(mean=1.0, m2=0.5) for name in METRIC_NAMES},
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("age", [17, 76, -1])
def test_cell_rejects_ages_outside_the_published_range(age: int) -> None:
    with pytest.raises(ReferenceStatsError):
        ReferenceCell(
            race="Asian",
            gender="male",
            age=age,
            count=3,
            moments={name: MetricMoments(mean=1.0, m2=0.5) for name in METRIC_NAMES},
        )


@pytest.mark.parametrize("count", [0, -1])
def test_cell_rejects_non_positive_count(count: int) -> None:
    """テスト 7: 負の `count`。件数 0 のセルは配布されない（§9.5）。"""
    with pytest.raises(ReferenceStatsError):
        ReferenceCell(
            race="Asian",
            gender="male",
            age=30,
            count=count,
            moments={name: MetricMoments(mean=1.0, m2=0.5) for name in METRIC_NAMES},
        )


def test_cell_rejects_missing_metrics() -> None:
    """指標が欠けたセルを受け取らない。欠測を 0 で埋めない（不変条件 11）。"""
    partial = {name: MetricMoments(mean=1.0, m2=0.5) for name in METRIC_NAMES[:-1]}

    with pytest.raises(ReferenceStatsError):
        ReferenceCell(race="Asian", gender="male", age=30, count=3, moments=partial)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_moments_reject_non_finite_values(bad: float) -> None:
    """テスト 7: NaN / Inf を受け取った時点で拒否する（不変条件 11）。"""
    with pytest.raises(ReferenceStatsError):
        MetricMoments(mean=bad, m2=0.5)
    with pytest.raises(ReferenceStatsError):
        MetricMoments(mean=1.0, m2=bad)


def test_moments_reject_negative_sum_of_squared_deviations() -> None:
    """テスト 7: `m2` は二乗偏差の和なので負になりえない。"""
    with pytest.raises(ReferenceStatsError):
        MetricMoments(mean=1.0, m2=-1e-9)


def test_filter_rejects_unknown_age_band_at_selection() -> None:
    with pytest.raises(ReferenceStatsError):
        select_cells(_population(), ReferenceFilter(age_bands=frozenset({"76-99"})))
