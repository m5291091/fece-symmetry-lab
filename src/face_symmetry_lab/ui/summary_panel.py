"""FR-021: セッション要約の描画。

算術は `domain.session_summary` が持ち、ここは整形と配置だけを担う。**FR-042 の
レポートも同じ domain の結果を読む**ので、片方だけが計算を持つことはない。

置き場所は「まとめ・エクスポート」タブである。5 Hz で再描画される詳細結果の
fragment へ入れると、最大 3,000 行の要約を毎秒 5 回計算することになる。要約は
セッション単位の成果物であり、現在フレームの表示とは更新の理由が違う。

**文言の制約**（§10.7）: これは 1 セッション内の記述統計であって、母集団に対する
順序への写像ではない。「偏差値」「順位」「パーセンタイル」「合否」「平均より上」「平均より下」
と価値判断語を使わない。四分位範囲は「変動幅（真ん中 50%）」と呼ぶ——「分位」は
§10.7 が順位化の語として名指ししているため、同じ語で別のものを指さない。

**FR-043**: `SUMMARY_ROW_LABELS` の表示側ラベルは `ui.readings.PART_LABELS`
（他エージェントの担当ファイル）を経由した値であり、この module から直接
`t()` へ渡せない（`t()` の引数は同じモジュールの module-level literal か文字列
リテラルでなければならない、`tests/ui/test_i18n.py`）。`summary_row_label` が
`key`（`"symmetry"` / `domain.metrics.PART_NAMES` の要素、いずれも言語に依存
しない固定識別子）から現在の表示言語のラベルを引き、`ui.report`（同じ訳表
`_report.py` を共有する）もこの関数越しに読む。

**「全体のスコア」は撤回した**（2026-09-03、§0 項目 34）。表の読みは
`"symmetry"` と 4 部位の 5 つだけになり、単位はすべて % で揃う。
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
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import PART_LABELS

SUMMARY_HEADING: Final[str] = "##### このセッションの要約"

#: §14.4 の実測。中央値だけを見せると「自分の値が決まった」と読まれるため、
#: 撮り直しで動くことを同じ場所へ置く。開かないと見えない場所へは置かない。
SUMMARY_LIMITS_NOTE: Final[str] = "撮り直すだけでも中央値は動きます。"

#: 区間を分ける理由を、表のすぐ隣に置く。分かれていること自体が読み手には
#: 唐突なので、「設定が違えば比べられない」という理由まで書く（ADR-0009 (c)）。
SUMMARY_SEGMENT_NOTE: Final[str] = "設定が違う区間の値は、同じ中央値へまとめられません。"

SUMMARY_SPREAD_LABEL: Final[str] = "変動幅（真ん中50%）"

#: FR-021 (5) / §10.11-2: **「表示できた割合」までとし、できなかった内訳を出さない。**
#: 分けると、per-frame では同一文言にして区別させなかったものが、セッション単位で
#: 復元できてしまう。ラベルも「範囲外だった割合」のような裏返しにしない。
REFERENCE_AGE_RATE_LABEL: Final[str] = "参考年齢を表示できたフレーム"

REFERENCE_AGE_RATE_HELP: Final[str] = (
    "顔を検出して年齢の判定を行ったフレームのうち、参考年齢を表示できた割合です。"
    "顔が写っていないフレームは数えていません。"
)

#: 表の行の並び（キーの列挙のためだけに使う）。画面に出る 5 つと同じ順に
#: そろえる。FR-042 のレポートも同じ並びを読む（画面と保存物で行の順が違わない
#: ように）。**2 番目の要素（日本語ラベル）は表示に使わない**——実際に出す
#: ラベルは必ず `summary_row_label(key)` を経由し、そちらが表示言語（FR-043）
#: へ訳す。ここへ英訳を足す必要はない。
SUMMARY_ROW_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("symmetry", "全体の対称度"),
    *tuple((name, PART_LABELS[name]) for name in PART_LABELS),
)

_VALID_RATE_LABEL: Final[str] = "計測条件を満たしたフレーム"
_VALID_RATE_HELP: Final[str] = "記録した全フレームのうち、品質ゲートを通った割合です。"
_VALID_SPAN_LABEL: Final[str] = "有効だった長さ"
_VALID_SPAN_HELP: Final[str] = (
    "最初と最後の有効フレームの間隔です。連続して有効だった時間ではありません。"
)
_HOLD_REASONS_CAPTION: Final[str] = "計測条件を満たさなかった主な理由"

_SPAN_SECONDS_TEMPLATE: Final[str] = "{value:.1f} 秒"
_RECORD_COUNTS_TEMPLATE: Final[str] = "記録 {total:,} 行 · うち有効 {ok:,} 行"
_REASON_LINE_TEMPLATE: Final[str] = "- {guidance}（{count:,} 回 · {share:.0%}）"
_SEGMENT_BREAKDOWN_EXPANDER_TEMPLATE: Final[str] = "区間ごとの内訳（{count} 区間）"
_SEGMENT_HEADING_WITH_ID_TEMPLATE: Final[str] = "**区間 {index}**（`{short}`）"
_SEGMENT_HEADING_TEMPLATE: Final[str] = "**区間 {index}**"


def summary_row_label(key: str) -> str:
    """`SUMMARY_ROW_LABELS` の鍵から、現在の表示言語での見出しを引く。

    値（`PART_LABELS[name]` の日本語）ではなく鍵で分岐するのは、鍵が言語に
    依存しない固定の識別子（`"symmetry"` / `domain.metrics.PART_NAMES` の要素）
    だからである。`ui.report` も同じ関数を読み、レポートと画面のラベルが
    ここ 1 か所からそろう。
    """
    if key == "symmetry":
        return t("全体の対称度")
    if key == "contour":
        return t("輪郭")
    if key == "eyes":
        return t("目")
    if key == "nose":
        return t("鼻")
    if key == "mouth":
        return t("口")
    return dict(SUMMARY_ROW_LABELS).get(key, key)


def format_summary_percent(value: float | None) -> str:
    """要約の対称度（既に % へ変換済みの値）。欠測は `—`（0 で埋めない）。

    `ui.readings.format_symmetry` とは入力が違う——あちらは距離を受け取って
    変換まで行うが、要約は `domain.session_summary` が変換済みの値を返す。
    桁は同じ 2 桁にそろえる（画面の別々の場所で丸めが違わないように）。
    """
    return "—" if value is None else f"{value:.2f}%"


def _cells(reading: SummaryReading | None) -> tuple[str, str, str]:
    """中央値・変動幅・最小〜最大の 3 セル。欠測は `—`（0 で埋めない）。"""
    if reading is None:
        return ("—", "—", "—")
    spread = format_summary_percent(reading.iqr) if reading.iqr is not None else "—"
    return (
        format_summary_percent(reading.median),
        spread,
        f"{format_summary_percent(reading.minimum)} 〜 {format_summary_percent(reading.maximum)}",
    )


def reading_for(section: SummarySection, key: str) -> SummaryReading | None:
    """`SUMMARY_ROW_LABELS` のキー 1 つに対応する読みを引く。"""
    return _reading_of(section, key)


def _reading_of(section: SummarySection, key: str) -> SummaryReading | None:
    if key == "symmetry":
        return section.symmetry
    return section.parts.get(key)


def _render_reading_table(section: SummarySection) -> None:
    """5 つの読みを Markdown 表で出す。

    **`st.metric` へ 5 つ並べない。** `st.metric` の値は 36px・`nowrap`・
    `text-overflow: ellipsis` なので、列を分けると幅が足りずに末尾が省略記号へ
    置き換わる（`ui.reference_page` が実測で確認済みの罠）。表なら列幅に合わせて
    縮む。`st.dataframe`/pandas を使わないのも同じ module の前例に従う。
    """
    header = (
        f"| {t('値')} | {t('中央値')} | {t(SUMMARY_SPREAD_LABEL)} | "
        f"{t('最小 〜 最大')} | {t('件数')} |"
    )
    divider = "|---|---:|---:|---:|---:|"
    rows = []
    for key, _label in SUMMARY_ROW_LABELS:
        label = summary_row_label(key)
        reading = reading_for(section, key)
        median, spread, extremes = _cells(reading)
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
    st.caption(t(_HOLD_REASONS_CAPTION))
    lines = []
    for code, count in section.reason_counts:
        share = count / section.total_count if section.total_count else 0.0
        guidance = t(REASON_GUIDANCE[code])
        lines.append(t(_REASON_LINE_TEMPLATE).format(guidance=guidance, count=count, share=share))
    st.markdown("\n".join(lines))


def _render_section(section: SummarySection) -> None:
    valid, span = st.columns(2)
    valid.metric(
        t(_VALID_RATE_LABEL),
        f"{section.valid_rate * 100.0:.1f}%",
        help=t(_VALID_RATE_HELP),
    )
    span_text = (
        "—"
        if section.valid_span_ms is None
        else t(_SPAN_SECONDS_TEMPLATE).format(value=section.valid_span_ms / 1000.0)
    )
    span.metric(t(_VALID_SPAN_LABEL), span_text, help=t(_VALID_SPAN_HELP))
    rate = section.reference_age_rate
    if rate is not None:
        # FR-021 (5): ゲートを呼んだセッションでだけ出す。呼んでいないセッション
        # へ 0% と書くと、測っていないことと「0% だった」が混ざる。
        st.metric(
            t(REFERENCE_AGE_RATE_LABEL),
            f"{rate * 100.0:.0f}%",
            help=t(REFERENCE_AGE_RATE_HELP),
        )
    st.caption(t(_RECORD_COUNTS_TEMPLATE).format(total=section.total_count, ok=section.ok_count))
    _render_reading_table(section)
    _render_reasons(section)


def render_session_summary(summary: SessionSummary) -> None:
    """FR-021 の 2 段（全体 → 区間ごと）を描画する。

    記録が 1 件も無いセッション（`overall` が `None`）では**何も描かない**。
    0% と 0.00 を並べると「測ったが 0 だった」と読めてしまう（不変条件 11）。
    """
    if summary.overall is None:
        return

    st.markdown(t(SUMMARY_HEADING))
    _render_section(summary.overall)
    st.caption(t(SUMMARY_LIMITS_NOTE))

    # 区間が 1 つだけのセッション（設定を変えなかった場合）は、全体と同じ表を
    # もう一度出すことになるので畳んでおく。
    if len(summary.segments) <= 1:
        return
    expander_title = t(_SEGMENT_BREAKDOWN_EXPANDER_TEMPLATE).format(count=len(summary.segments))
    with st.expander(expander_title):
        st.caption(t(SUMMARY_SEGMENT_NOTE))
        for index, section in enumerate(summary.segments, start=1):
            # 区間の見出しは通し番号で出す。`segment_id` は UUID なので人には
            # 読めないが、**先頭 8 桁だけは添える**——エクスポートした CSV の
            # `segment_id` 列と突き合わせる唯一の手掛かりになる（`ui.export_panel`
            # が config fingerprint を先頭 12 桁で示すのと同じ作法）。
            short = (section.segment_id or "")[:8]
            heading = (
                t(_SEGMENT_HEADING_WITH_ID_TEMPLATE).format(index=index, short=short)
                if short
                else t(_SEGMENT_HEADING_TEMPLATE).format(index=index)
            )
            st.markdown(heading)
            _render_section(section)
