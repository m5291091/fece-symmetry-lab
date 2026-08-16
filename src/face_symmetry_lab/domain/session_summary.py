"""FR-021: 1 セッションの記述統計（純粋計算）。

3 分のセッションは 1,400 行になる。人が読める結果にするには要約が要るが、
**要約は記述統計以上のことを主張してはならない**（§10.7）。ここが出すのは 4 つ
だけである——有効率・中央値・変動幅・主な HOLD 理由。代表値や総合評価は作らない。

**`segment_id` ごとと全体の 2 段で返す。** `segment_id` はセッションのリセット・
入力切替・指標に影響する設定変更（平滑化時間の変更を含む、FR-040）で更新される。
異なる `ema_tau_seconds` で測った区間の値を同じ中央値へ混ぜると、どの設定の測定
とも言えない数になる（ADR-0009 (c)）。

**読むのは生値の列**（`distance3d_*_raw`）であって平滑値ではない。平滑値は
`ema_tau_seconds` に依存するため、同じ撮影でも表示設定を変えると要約が動いて
しまう。加えて EMA は連続する行を強く相関させるので、その中央値は「分布の中心」
という意味を失う。

**この module は何も描画しない。** 画面（詳細結果タブ）と FR-042 のレポートは
どちらもここを読む。片方だけが計算を持つと、同じセッションに 2 つの中央値が
生まれる。
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import isfinite

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import ReasonCode, Status
from face_symmetry_lab.domain.score_scale import score_points, symmetry_percent

#: FR-021: 画面へ出す HOLD 理由の件数。上位これだけで、利用者が次に何をすべきか
#: は決まる。全件を並べても行動は変わらない。
TOP_REASON_COUNT: int = 3

#: 四分位を出すのに要る最小の標本数。`n=1` では四分位そのものが定義できないため
#: 欠測にする（`domain.reference_stats` が `n=1` で標本標準偏差を欠測にするのと
#: 同じ作法）。0 で埋めない（不変条件 11）。
MIN_SAMPLES_FOR_SPREAD: int = 2


class SummaryError(ValueError):
    """要約の入力が不正である。"""


@dataclass(frozen=True, slots=True)
class SummaryReading:
    """1 つの測定値に対する分布の記述。単位は**表示単位**である。

    対称度は %、全体のスコアは点。変換は単調なので順序と中央値は変換の前後で
    一致するが、四分位は補間を伴い、clamp を含むスコアの目盛りでは一致しない。
    そのため**必ず変換後の値で計算する**（AC-SUMMARY-01）。

    対称度は距離の最大が % の最小へ写るため、`minimum` / `maximum` は距離で見た
    ときと入れ替わる。
    """

    median: float
    minimum: float
    maximum: float
    #: 四分位範囲（Q3 − Q1）。標本が 1 件のときは欠測。
    iqr: float | None
    #: この値を持っていた OK 行の数。部位が個別に欠測したフレームがあると、
    #: 同じ区間でも読みごとに違う数になる。
    sample_count: int


@dataclass(frozen=True, slots=True)
class SummarySection:
    """1 つの区間（`segment_id` 単位、または全体）の要約。"""

    #: 全体の段では `None`。
    segment_id: str | None
    total_count: int
    ok_count: int
    #: OK 行 ÷ 記録された全行数。**OK が 0 件でも 0.0 であり欠測ではない**——
    #: 分母は記録された行数であり、確定しているためである（AC-SUMMARY-02）。
    valid_rate: float
    #: 最初と最後の OK 行の `elapsed_ms` の差。OK 行が 2 件未満なら欠測。
    valid_span_ms: int | None
    symmetry: SummaryReading | None
    score: SummaryReading | None
    parts: dict[str, SummaryReading | None]
    #: 出現回数の多い順、上位 `TOP_REASON_COUNT` 件。同数のときは最初に現れた順。
    reason_counts: tuple[tuple[ReasonCode, int], ...]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """FR-021 の 2 段。記録が 1 件も無ければ `overall` は `None`。"""

    overall: SummarySection | None
    segments: tuple[SummarySection, ...]


def _distance(record: object, field: str) -> float | None:
    """生値の列を 1 つ読み、壊れた値をここで弾く。

    非有限値・負値を黙って中央値へ混ぜない（不変条件 11）。`SessionRecord` 側の
    pydantic 検証を通っていれば起こらないが、要約は「記録された列を読むだけ」の
    立場なので、自分でも確かめる。
    """
    value = getattr(record, field, None)
    if value is None:
        return None
    number = float(value)
    if not isfinite(number) or number < 0.0:
        raise SummaryError(f"{field} は 0 以上の有限値である必要があります。")
    return number


def _reading(
    distances: Sequence[float],
    *,
    convert: Callable[[float | None], float | None],
) -> SummaryReading | None:
    """距離の列を表示単位へ変換してから、中央値・最小最大・四分位を作る。"""
    converted = [value for value in map(convert, distances) if value is not None]
    if not converted:
        return None
    iqr: float | None = None
    if len(converted) >= MIN_SAMPLES_FOR_SPREAD:
        first, _median, third = statistics.quantiles(converted, n=4, method="inclusive")
        iqr = third - first
    return SummaryReading(
        median=statistics.median(converted),
        minimum=min(converted),
        maximum=max(converted),
        iqr=iqr,
        sample_count=len(converted),
    )


def _collect(records: Sequence[object], field: str) -> list[float]:
    """OK 行のうち、その列を持っている行の値を集める。

    **`analysis_status` を必ず見る。** 値の有無だけで拾うと、ゲートが閉じた行に
    残っていた値が混ざる。
    """
    values: list[float] = []
    for record in records:
        if getattr(record, "analysis_status", None) is not Status.OK:
            continue
        value = _distance(record, field)
        if value is not None:
            values.append(value)
    return values


def _section(records: Sequence[object], *, segment_id: str | None) -> SummarySection:
    ok_rows = [
        record for record in records if getattr(record, "analysis_status", None) is Status.OK
    ]
    elapsed = sorted(int(getattr(record, "elapsed_ms", 0)) for record in ok_rows)
    span = elapsed[-1] - elapsed[0] if len(elapsed) >= MIN_SAMPLES_FOR_SPREAD else None

    order: dict[ReasonCode, int] = {}
    counter: Counter[ReasonCode] = Counter()
    for record in records:
        # 1 行に複数コードがあれば、各々を 1 件として数える（AC-SUMMARY-01）。
        for code in getattr(record, "reason_codes", ()) or ():
            order.setdefault(code, len(order))
            counter[code] += 1
    ranked = sorted(counter.items(), key=lambda item: (-item[1], order[item[0]]))

    return SummarySection(
        segment_id=segment_id,
        total_count=len(records),
        ok_count=len(ok_rows),
        valid_rate=len(ok_rows) / len(records) if records else 0.0,
        valid_span_ms=span,
        symmetry=_reading(_collect(records, "distance3d_overall_raw"), convert=symmetry_percent),
        score=_reading(_collect(records, "distance3d_parts_equal_raw"), convert=score_points),
        parts={
            name: _reading(_collect(records, f"distance3d_{name}_raw"), convert=symmetry_percent)
            for name in PART_NAMES
        },
        reason_counts=tuple(ranked[:TOP_REASON_COUNT]),
    )


#: FR-042 の推移図が描く最大の点数。3,000 点を SVG の `points` 属性へ並べても
#: 720px の幅では 1 点あたり 0.24px にしかならず、読めないままファイルだけが
#: 太る。間引きは表示専用で、要約の統計には一切使わない。
MAX_TREND_POINTS: int = 600


def trend_series(
    records: Iterable[object],
    *,
    max_points: int = MAX_TREND_POINTS,
) -> tuple[float, ...]:
    """FR-042 §9.6.1 の推移図が描く系列（対称度 %）。

    要約と同じく **OK 行の生値だけ**を読み、表示単位へ変換して返す。`max_points`
    を超えるときは等間隔で間引く——**先頭や末尾を切り捨てない**。切り捨てると、
    セッションの一部だけを描いた図に「全体の推移」という見出しが付く。
    """
    values = [
        percent
        for percent in (
            symmetry_percent(value) for value in _collect(list(records), "distance3d_overall_raw")
        )
        if percent is not None
    ]
    if max_points < 1:
        raise SummaryError("推移図の点数は 1 以上である必要があります。")
    if len(values) <= max_points:
        return tuple(values)
    stride = len(values) / max_points
    return tuple(values[min(len(values) - 1, int(index * stride))] for index in range(max_points))


def summarize_session(records: Iterable[object]) -> SessionSummary:
    """FR-021 の要約を、`segment_id` ごとと全体の 2 段で返す。

    ``records`` は `domain.export.SessionRecord` の列を想定するが、読むのは
    `analysis_status` / `segment_id` / `elapsed_ms` / `reason_codes` と 6 つの
    生値の列だけである。約 50 列のうち要約に関係するのはこれだけなので、
    型ではなく読む属性で結んである。

    記録が 1 件も無ければ `overall` は `None` を返す。0 件のセッションへ 0% や
    0.00 を並べると、「測ったが 0 だった」と読めてしまう（不変条件 11）。
    """
    rows = list(records)
    if not rows:
        return SessionSummary(overall=None, segments=())

    grouped: dict[str, list[object]] = {}
    for record in rows:
        # 出現順を保つ（Python の dict は挿入順）。時間順に読める並びにするため
        # であり、辞書順や set の反復順に頼ると並びが不定になる。
        grouped.setdefault(str(getattr(record, "segment_id", "")), []).append(record)

    return SessionSummary(
        overall=_section(rows, segment_id=None),
        segments=tuple(
            _section(group, segment_id=segment_id) for segment_id, group in grouped.items()
        ),
    )
