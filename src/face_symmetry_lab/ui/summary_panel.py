"""FR-021: セッション要約の描画。

算術は `domain.session_summary` が持ち、ここは整形と配置だけを担う。**FR-042 の
レポートも同じ domain の結果を読む**ので、片方だけが計算を持つことはない。

置き場所は「まとめ・エクスポート」タブである。5 Hz で再描画される詳細結果の
fragment へ入れると、最大 3,000 行の要約を毎秒 5 回計算することになる。要約は
セッション単位の成果物であり、現在フレームの表示とは更新の理由が違う。

**文言の制約**（§10.7）: これは 1 セッション内の記述統計であって、母集団に対する
順序への写像ではない。「偏差値」「順位」「パーセンタイル」「合否」「平均より上」「平均より下」
と価値判断語を使わない。四分位範囲は「変動幅（真ん中 50%）」と呼ぶ——「分位」は
§10.12 が順位化の語として名指ししているため、同じ語で別のものを指さない。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.domain.session_summary import (
    SessionSummary,
    SummaryReading,
    SummarySection,
)
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.readings import PART_LABELS

SUMMARY_HEADING: Final[str] = "##### このセッションの要約"

#: §14.4 の実測（3 セッションの全幅 6.8 点）。中央値だけを見せると「自分の値が
#: 決まった」と読まれるため、撮り直しで動く量を同じ場所へ置く。`ui.scores` の
#: `SCORE_SCALE_CAPTION` と同じ役割で、開かないと見えない場所へは置かない。
SUMMARY_LIMITS_NOTE: Final[str] = (
    "同一人物・同一環境でも、撮り直すだけで中央値は数ポイント動きます"
    "（実測で全幅 6.8 点）。ここにあるのはこの 1 セッションの記述統計で、"
    "他の人との比較にも、医学的な評価にも使えません。"
)

#: 区間を分ける理由を、表のすぐ隣に置く。分かれていること自体が読み手には
#: 唐突なので、「設定が違えば比べられない」という理由まで書く（ADR-0009 (c)）。
SUMMARY_SEGMENT_NOTE: Final[str] = (
    "セッションのリセットや平滑化時間の変更で区間が分かれます。"
    "設定が違う区間の値は、同じ中央値へまとめられません。"
)

SUMMARY_SPREAD_LABEL: Final[str] = "変動幅（真ん中50%）"

#: 表の行の並び。画面に出る 6 つと同じ順・同じラベルにそろえる。
_ROW_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("symmetry", "全体の対称度"),
    ("score", "全体のスコア"),
    *tuple((name, PART_LABELS[name]) for name in PART_LABELS),
)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


def _points(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} 点"


def _cells(reading: SummaryReading | None, *, points: bool) -> tuple[str, str, str]:
    """中央値・変動幅・最小〜最大の 3 セル。欠測は `—`（0 で埋めない）。"""
    fmt = _points if points else _percent
    if reading is None:
        return ("—", "—", "—")
    spread = fmt(reading.iqr) if reading.iqr is not None else "—"
    return (fmt(reading.median), spread, f"{fmt(reading.minimum)} 〜 {fmt(reading.maximum)}")


def _reading_of(section: SummarySection, key: str) -> SummaryReading | None:
    if key == "symmetry":
        return section.symmetry
    if key == "score":
        return section.score
    return section.parts.get(key)


def _render_reading_table(section: SummarySection) -> None:
    """6 つの読みを Markdown 表で出す。

    **`st.metric` へ 6 つ並べない。** `st.metric` の値は 36px・`nowrap`・
    `text-overflow: ellipsis` なので、列を分けると幅が足りずに末尾が省略記号へ
    置き換わる（`ui.reference_page` が実測で確認済みの罠）。表なら列幅に合わせて
    縮む。`st.dataframe`/pandas を使わないのも同じ module の前例に従う。
    """
    header = f"| 値 | 中央値 | {SUMMARY_SPREAD_LABEL} | 最小 〜 最大 | 件数 |"
    divider = "|---|---:|---:|---:|---:|"
    rows = []
    for key, label in _ROW_LABELS:
        reading = _reading_of(section, key)
        median, spread, extremes = _cells(reading, points=key == "score")
        count = "—" if reading is None else f"{reading.sample_count:,}"
        rows.append(f"| {label} | {median} | {spread} | {extremes} | {count} |")
    st.markdown("\n".join((header, divider, *rows)))


def _render_reasons(section: SummarySection) -> None:
    """主な HOLD 理由。**コードではなく §6.3 の指示文で見せる。**

    `REASON_GUIDANCE` は品質ゲートが閉じている間に詳細結果へ出るのと同じ表で、
    要約のためだけに 2 つ目の文言表を作らない（AC-GUIDE-01 と同じ辞書）。
    """
    if not section.reason_counts:
        return
    st.caption("計測条件を満たさなかった主な理由")
    lines = []
    for code, count in section.reason_counts:
        share = count / section.total_count if section.total_count else 0.0
        guidance = REASON_GUIDANCE.get(code, code.value)
        lines.append(f"- {guidance}（{count:,} 回 · {share:.0%}）")
    st.markdown("\n".join(lines))


def _render_section(section: SummarySection) -> None:
    valid, span = st.columns(2)
    valid.metric(
        "計測条件を満たしたフレーム",
        f"{section.valid_rate * 100.0:.1f}%",
        help="記録した全フレームのうち、品質ゲートを通った割合です。値の良し悪しではありません。",
    )
    span.metric(
        "有効だった長さ",
        "—" if section.valid_span_ms is None else f"{section.valid_span_ms / 1000.0:.1f} 秒",
        help="最初と最後の有効フレームの間隔です。連続して有効だった時間ではありません。",
    )
    st.caption(f"記録 {section.total_count:,} 行 · うち有効 {section.ok_count:,} 行")
    _render_reading_table(section)
    _render_reasons(section)


def render_session_summary(summary: SessionSummary) -> None:
    """FR-021 の 2 段（全体 → 区間ごと）を描画する。

    記録が 1 件も無いセッション（`overall` が `None`）では**何も描かない**。
    0% と 0.00 を並べると「測ったが 0 だった」と読めてしまう（不変条件 11）。
    """
    if summary.overall is None:
        return

    st.markdown(SUMMARY_HEADING)
    _render_section(summary.overall)
    st.caption(SUMMARY_LIMITS_NOTE)

    # 区間が 1 つだけのセッション（設定を変えなかった場合）は、全体と同じ表を
    # もう一度出すことになるので畳んでおく。
    if len(summary.segments) <= 1:
        return
    with st.expander(f"区間ごとの内訳（{len(summary.segments)} 区間）"):
        st.caption(SUMMARY_SEGMENT_NOTE)
        for index, section in enumerate(summary.segments, start=1):
            # 区間の見出しは通し番号で出す。`segment_id` は UUID なので人には
            # 読めないが、**先頭 8 桁だけは添える**——エクスポートした CSV の
            # `segment_id` 列と突き合わせる唯一の手掛かりになる（`ui.export_panel`
            # が config fingerprint を先頭 12 桁で示すのと同じ作法）。
            short = (section.segment_id or "")[:8]
            st.markdown(f"**区間 {index}**（`{short}`）" if short else f"**区間 {index}**")
            _render_section(section)
