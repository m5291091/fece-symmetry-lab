"""FR-021: セッション要約の純粋計算（`domain.session_summary`）。

守るのは 3 点である。

1. **記述統計以上のことを主張しない。** 出るのは有効率・中央値・変動幅・HOLD 理由
   だけで、代表値や総合評価を作らない（§10.7）。
2. **設定が違う区間の値を混ぜない。** `segment_id` ごとに分けるのは、異なる
   `ema_tau_seconds` で測った区間の中央値が「どの設定の測定でもない数」になる
   ためである（ADR-0009 (c)）。
3. **読むのは生値の列。** 平滑値を読む実装は、表示設定を変えただけで要約が動く
   （FR-040 が「測定値を変えない」と言っているのに要約だけ変わる）。
"""

from __future__ import annotations

import statistics

import pytest

from face_symmetry_lab.domain.models import ReasonCode, Status
from face_symmetry_lab.domain.session_summary import (
    SummaryError,
    summarize_session,
    trend_series,
)


class _Record:
    """要約が読む列だけを持つ最小の記録。

    実物の `SessionRecord` は約 50 列あり、そのほとんどは要約と関係がない。
    ここで小さな stub を使うのは、**要約が実際に読んでいる列を明示するため**で
    もある（stub に無い列を読み始めたらここで落ちる）。
    """

    def __init__(
        self,
        *,
        status: Status = Status.OK,
        segment_id: str = "seg-1",
        elapsed_ms: int = 0,
        reason_codes: tuple[ReasonCode, ...] = (),
        overall: float | None = 0.02,
        equal_weight: float | None = 0.03,
        contour: float | None = 0.01,
        eyes: float | None = 0.02,
        nose: float | None = 0.03,
        mouth: float | None = 0.015,
        smooth: float | None = 0.5,
    ) -> None:
        self.analysis_status = status
        self.segment_id = segment_id
        self.elapsed_ms = elapsed_ms
        self.reason_codes = reason_codes
        self.distance3d_overall_raw = overall
        self.distance3d_parts_equal_raw = equal_weight
        self.distance3d_contour_raw = contour
        self.distance3d_eyes_raw = eyes
        self.distance3d_nose_raw = nose
        self.distance3d_mouth_raw = mouth
        #: 罠用: 生値と大きく離れた平滑値を入れてある。要約がこちらを読んだら
        #: 中央値がまったく違う値になり、下の期待値と合わない。
        self.distance3d_overall_smooth = smooth


def _ok_records(overalls: list[float], **kwargs) -> list[_Record]:
    return [
        _Record(overall=value, elapsed_ms=index * 100, **kwargs)
        for index, value in enumerate(overalls)
    ]


# --- 有効率 -------------------------------------------------------------------


def test_the_valid_rate_is_ok_rows_over_every_recorded_row() -> None:
    records = [
        _Record(status=Status.OK),
        _Record(status=Status.OK),
        _Record(status=Status.HOLD, overall=None),
        _Record(status=Status.NO_FACE, overall=None),
    ]

    summary = summarize_session(records)

    assert summary.overall.valid_rate == pytest.approx(0.5)
    assert summary.overall.ok_count == 2
    assert summary.overall.total_count == 4


def test_a_session_without_any_valid_frame_has_a_zero_rate_not_a_missing_one() -> None:
    """有効率の分母は記録された行数であり、確定している（AC-SUMMARY-02）。"""
    summary = summarize_session([_Record(status=Status.HOLD, overall=None)])

    assert summary.overall.valid_rate == pytest.approx(0.0)
    assert summary.overall.symmetry is None


def test_an_empty_session_produces_nothing_rather_than_zeros() -> None:
    summary = summarize_session([])

    assert summary.overall is None
    assert summary.segments == ()


# --- 中央値と変動幅 -----------------------------------------------------------


def test_the_median_matches_a_direct_calculation_over_the_same_rows() -> None:
    values = [0.010, 0.020, 0.030, 0.040, 0.050]
    summary = summarize_session(_ok_records(values))

    reading = summary.overall.symmetry
    assert reading is not None
    # 表示単位（対称度 %）で出る。変換は単調なので順序統計量と可換である。
    assert reading.median == pytest.approx(100.0 * (1.0 - statistics.median(values)))


def test_the_summary_reads_the_raw_column_and_not_the_smoothed_one() -> None:
    """罠: 平滑値を読む実装は、表示設定を変えただけで要約が動く（FR-040）。

    stub の平滑値は生値から大きく離してあるので、こちらを読むと落ちる。
    """
    summary = summarize_session(_ok_records([0.02, 0.02, 0.02], smooth=0.5))

    reading = summary.overall.symmetry
    assert reading is not None
    assert reading.median == pytest.approx(98.0)


def test_the_spread_is_the_interquartile_range_and_the_extremes() -> None:
    values = [0.010, 0.020, 0.030, 0.040]
    summary = summarize_session(_ok_records(values))

    reading = summary.overall.symmetry
    assert reading is not None
    # 対称度へ変換すると大小が反転するため、最小と最大は距離の最大・最小から来る。
    assert reading.minimum == pytest.approx(100.0 * (1.0 - max(values)))
    assert reading.maximum == pytest.approx(100.0 * (1.0 - min(values)))
    assert reading.iqr is not None
    assert reading.iqr > 0.0


def test_the_interquartile_range_is_missing_for_a_single_sample() -> None:
    """`domain.reference_stats` が `n=1` で標本標準偏差を欠測にするのと同じ作法。"""
    summary = summarize_session(_ok_records([0.02]))

    reading = summary.overall.symmetry
    assert reading is not None
    assert reading.median == pytest.approx(98.0)
    assert reading.iqr is None


def test_every_displayed_reading_is_summarised() -> None:
    """画面に出る 6 つ（全体の対称度・全体のスコア・部位別 4 つ）を過不足なく持つ。"""
    summary = summarize_session(_ok_records([0.02, 0.03]))

    assert summary.overall.symmetry is not None
    assert summary.overall.score is not None
    assert set(summary.overall.parts) == {"contour", "eyes", "nose", "mouth"}
    assert all(reading is not None for reading in summary.overall.parts.values())


def test_the_score_is_summarised_on_the_points_scale() -> None:
    """§8.6.7: スコアは点。対称度 % と同じ目盛りに見えてはならない。"""
    summary = summarize_session(_ok_records([0.02], equal_weight=0.03))

    score = summary.overall.score
    assert score is not None
    # `score_points` のアンカー（D_FULL=0.010 / D_ZERO=0.050）から 0.03 は 50 点。
    assert score.median == pytest.approx(50.0)


def test_a_reading_missing_on_some_rows_uses_only_the_rows_that_have_it() -> None:
    """部位が個別に欠測したフレームがあっても、他の部位の中央値は出続ける。"""
    records = [
        _Record(overall=0.02, nose=None),
        _Record(overall=0.02, nose=0.03),
        _Record(overall=0.02, nose=0.05),
    ]

    summary = summarize_session(records)

    assert summary.overall.symmetry is not None
    nose = summary.overall.parts["nose"]
    assert nose is not None
    assert nose.median == pytest.approx(100.0 * (1.0 - 0.04))


def test_a_reading_missing_on_every_row_is_missing_rather_than_zero() -> None:
    """不変条件 11: 欠測は 0 で埋めない。"""
    summary = summarize_session(_ok_records([0.02], equal_weight=None))

    assert summary.overall.score is None
    assert summary.overall.symmetry is not None


def test_rows_that_are_not_ok_never_contribute_a_value() -> None:
    """罠: status を見ずに値の有無だけで拾う実装は、HOLD の残骸を混ぜる。"""
    records = [
        _Record(status=Status.OK, overall=0.02),
        _Record(status=Status.HOLD, overall=0.40),
    ]

    summary = summarize_session(records)

    reading = summary.overall.symmetry
    assert reading is not None
    assert reading.median == pytest.approx(98.0)
    assert reading.sample_count == 1


# --- HOLD 理由 ----------------------------------------------------------------


def test_reason_codes_are_counted_per_code_not_per_row() -> None:
    """1 行に複数コードがあれば、各々を 1 件として数える（AC-SUMMARY-01）。"""
    records = [
        _Record(status=Status.HOLD, overall=None, reason_codes=(ReasonCode.OFF_AXIS_YAW,)),
        _Record(
            status=Status.HOLD,
            overall=None,
            reason_codes=(ReasonCode.OFF_AXIS_YAW, ReasonCode.TOO_FAR),
        ),
    ]

    summary = summarize_session(records)

    counts = dict(summary.overall.reason_counts)
    assert counts[ReasonCode.OFF_AXIS_YAW] == 2
    assert counts[ReasonCode.TOO_FAR] == 1


def test_only_the_three_most_common_reasons_are_kept() -> None:
    """§10.7 の趣旨とは別に、上位 3 件で足りる（FR-021）。"""
    codes = (
        ReasonCode.OFF_AXIS_YAW,
        ReasonCode.OFF_AXIS_PITCH,
        ReasonCode.OFF_AXIS_ROLL,
        ReasonCode.TOO_FAR,
    )
    records = [
        _Record(status=Status.HOLD, overall=None, reason_codes=(code,) * (index + 1))
        for index, code in enumerate(codes)
    ]

    summary = summarize_session(records)

    assert len(summary.overall.reason_counts) == 3
    # 多い順に並ぶ。
    assert [count for _, count in summary.overall.reason_counts] == [4, 3, 2]


def test_reason_shares_are_taken_over_every_recorded_row() -> None:
    records = [
        _Record(status=Status.OK),
        _Record(status=Status.HOLD, overall=None, reason_codes=(ReasonCode.TOO_FAR,)),
    ]

    summary = summarize_session(records)

    _, count = summary.overall.reason_counts[0]
    assert count == 1
    assert summary.overall.total_count == 2


# --- segment ごとの分割 -------------------------------------------------------


def test_segments_are_summarised_separately_and_never_mixed() -> None:
    """ADR-0009 (c): 設定が違う区間の値は同じ中央値へ混ざらない。

    罠: `segment_id` を無視する実装は、両区間を合わせた中央値 0.03 を返す。
    """
    records = [
        *_ok_records([0.02, 0.02], segment_id="seg-1"),
        *_ok_records([0.04, 0.04], segment_id="seg-2"),
    ]

    summary = summarize_session(records)

    assert [segment.segment_id for segment in summary.segments] == ["seg-1", "seg-2"]
    first, second = summary.segments
    assert first.symmetry is not None
    assert second.symmetry is not None
    assert first.symmetry.median == pytest.approx(98.0)
    assert second.symmetry.median == pytest.approx(96.0)


def test_segments_keep_the_order_they_first_appeared_in() -> None:
    """時間順に読める並びにする。罠: 辞書順や set の反復順に頼る実装は不定になる。"""
    records = [
        *_ok_records([0.02], segment_id="zzz"),
        *_ok_records([0.02], segment_id="aaa"),
    ]

    summary = summarize_session(records)

    assert [segment.segment_id for segment in summary.segments] == ["zzz", "aaa"]


def test_the_overall_tier_spans_every_segment() -> None:
    records = [
        *_ok_records([0.02, 0.02], segment_id="seg-1"),
        *_ok_records([0.04, 0.04], segment_id="seg-2"),
    ]

    summary = summarize_session(records)

    assert summary.overall.total_count == 4
    assert summary.overall.symmetry is not None
    assert summary.overall.symmetry.median == pytest.approx(97.0)


def test_the_elapsed_span_covers_the_valid_rows_only() -> None:
    records = [
        _Record(status=Status.HOLD, overall=None, elapsed_ms=0),
        _Record(status=Status.OK, elapsed_ms=1_000),
        _Record(status=Status.OK, elapsed_ms=4_000),
        _Record(status=Status.NO_FACE, overall=None, elapsed_ms=9_000),
    ]

    summary = summarize_session(records)

    assert summary.overall.valid_span_ms == 3_000


def test_a_span_needs_two_valid_rows() -> None:
    summary = summarize_session([_Record(status=Status.OK, elapsed_ms=1_000)])

    assert summary.overall.valid_span_ms is None


# --- 入力の拒否 ---------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_a_non_finite_or_negative_distance_is_refused(bad: float) -> None:
    """不変条件 11: 壊れた値を黙って中央値へ混ぜない。"""
    with pytest.raises(SummaryError):
        summarize_session([_Record(overall=bad)])


# --- FR-042: 推移図の系列 -----------------------------------------------------


def test_the_trend_series_carries_the_valid_rows_in_display_units() -> None:
    series = trend_series(_ok_records([0.010, 0.020, 0.030]))

    assert series == pytest.approx((99.0, 98.0, 97.0))


def test_the_trend_series_skips_rows_that_are_not_ok() -> None:
    records = [
        _Record(status=Status.OK, overall=0.02),
        _Record(status=Status.HOLD, overall=None),
        _Record(status=Status.OK, overall=0.03),
    ]

    assert trend_series(records) == pytest.approx((98.0, 97.0))


def test_a_long_session_is_thinned_without_losing_its_ends() -> None:
    """罠: 先頭 600 点で切る実装は、セッションの一部だけを「全体の推移」と呼ぶ。"""
    values = [0.010 + index * 0.00001 for index in range(3000)]

    series = trend_series(_ok_records(values), max_points=600)

    assert len(series) == 600
    assert series[0] == pytest.approx(100.0 * (1.0 - values[0]))
    # 末尾付近まで到達している（先頭だけを切り出していない）。
    assert series[-1] == pytest.approx(100.0 * (1.0 - values[-1]), abs=0.01)


def test_a_short_session_is_left_alone() -> None:
    series = trend_series(_ok_records([0.02, 0.03]), max_points=600)

    assert len(series) == 2


def test_an_empty_session_has_no_series() -> None:
    assert trend_series([]) == ()
