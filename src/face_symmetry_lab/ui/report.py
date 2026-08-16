"""FR-042 / §9.6: 人が読む結果レポート（単一 HTML）。

CSV / JSON（§9.2〜§9.4）は再現のための台帳で、列は機械可読な識別子である。
こちらは読み物であり、両者は目的が違うので併存させる。台帳側のスキーマは
この module によって一切変わらない。

**この module は Streamlit を呼ばない。** 受け取った値から HTML 文字列を組み立て
て返すだけなので、`AppTest` を使わずに文字列として検証できる。ダウンロードの
提示は `ui.export_panel` が行う。

**設計の中心は「保存されたファイルは画面の文脈を持たない」ことである。**
画面では周りの caption・タブ・About が担っていた注意書きが、ファイル 1 つの中に
揃っていなければ、数字だけが独り歩きする（FR-039 で併記の但し書きを同じ吹き出し
へ置いたのと同じ理由）。したがって §5.3 の留保・§10.7 の禁止用途・再現に必要な
版一式を、**必ず本体へ入れる**。

**外部リソースを参照しない**（AC-REPORT-02）。CSS は inline、図は `data:` URI。
開いた先で通信が起きるレポートは、ローカル完結という前提をファイル単体では
守れていない。

文言は既存の定数を再利用する（`ui.scores` の留保、`ui.readings` のフォーマッタ、
`ui.summary_panel` の要約の整形）。レポートだけが別の言い方・別の丸め方をすると、
画面と保存物で違うことを言うことになる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Final

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.reference_baseline import BaselineSummary
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.reference_baseline import (
    OVERALL_METRIC,
    PART_METRICS,
    SCORE_METRIC,
    baseline_reading,
)
from face_symmetry_lab.domain.score_explanation import explain_score
from face_symmetry_lab.domain.session_summary import (
    SessionSummary,
    SummarySection,
    TrendSeries,
)
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.readings import (
    GENDER_LABELS,
    PART_LABELS,
    baseline_note,
    format_baseline_score,
    format_baseline_symmetry,
    format_score,
    format_symmetry,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
    score_points_value,
    symmetry_percent,
)
from face_symmetry_lab.ui.scores import PART_EQUAL_WEIGHT_LABEL
from face_symmetry_lab.ui.summary_panel import (
    SUMMARY_LIMITS_NOTE,
    SUMMARY_ROW_LABELS,
    SUMMARY_SEGMENT_NOTE,
    SUMMARY_SPREAD_LABEL,
    format_summary_percent,
    format_summary_points,
    reading_for,
)

#: §9.6: レポートそのものの版。レコードスキーマ（`4.0.0`）とは別に数える——
#: 台帳の列が変わらなくても、レポートの構成は変わりうる。
REPORT_VERSION: Final[str] = "face-symmetry-report/1.0.0"

#: §5.3.1: 画面と同じく、レポートの注意も最小限にする。以前は §5.3 の留保全文と
#: §10.7 の禁止用途一覧を書き写しており、注意文が測定値より長かった——**読まれない
#: 注意文は、書いていないのと同じである。** 根拠と全文は About が持つ。
#:
#: ここへ残すのは、ファイル単体で成立させるために要る 2 文だけである。1 つ目は
#: 「この数字が何でないか」、2 つ目は使ってはいけない用途。どちらも受け取った人が
#: 画面を見ずに読む可能性がある。
REPORT_NOTES: Final[tuple[str, ...]] = (
    "同一人物・同一環境の中で傾向を見るための探索値です。他の人との比較、医療・採用・保険などの判断には使えません。",
    "詳しい算出方法と制限は、アプリの About ページに記載しています。",
)

#: §10.4 / ADR-0009 (d): 図を入れたときだけ足す 1 行。Camera のレポートは図を
#: 持たないので、この文も出ない。
REPORT_SHARING_NOTE: Final[str] = (
    "このファイルには顔写真が含まれます。保存後の共有先はアプリからは制御できません。"
)

_TREND_WIDTH_PX: Final[int] = 720
_TREND_HEIGHT_PX: Final[int] = 180

#: 推移図の縦軸。HUD（`ui.hud`）と同じ窓にそろえる——同じ測定の推移が、画面と
#: 保存物で別の縦幅に見えないようにするためである。
_TREND_Y_MIN: Final[float] = 95.0
_TREND_Y_MAX: Final[float] = 100.0

#: 「全体のスコア」の窓。§8.6.7 の目盛りは 0〜100 点で、実測はその全域へ広がる
#: （§14.4 の 3 セッションで 81〜88 点）ため、対称度と違って全域を見せる。
_SCORE_Y_MIN: Final[float] = 0.0
_SCORE_Y_MAX: Final[float] = 100.0

_STYLE: Final[str] = (
    "body{font-family:system-ui,-apple-system,'Hiragino Sans','Noto Sans JP',sans-serif;"
    "line-height:1.7;color:#1b1d21;background:#fff;margin:0 auto;padding:32px 20px;"
    "max-width:820px;font-size:15px}"
    "h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:32px 0 8px;"
    "border-bottom:1px solid #d8dbe0;padding-bottom:4px}h3{font-size:.95rem;margin:20px 0 6px}"
    "table{border-collapse:collapse;width:100%;margin:8px 0;font-variant-numeric:tabular-nums}"
    "th,td{border:1px solid #d8dbe0;padding:6px 10px;text-align:right}"
    "th:first-child,td:first-child{text-align:left}"
    "thead th{background:#f4f5f7;font-weight:600}"
    "figure{margin:16px 0}"
    "img{max-width:100%;height:auto;border:1px solid #d8dbe0;border-radius:6px}"
    "figcaption{font-size:.85rem;color:#4a4f57;margin-top:4px}"
    ".meta{font-size:.8rem;color:#4a4f57}"
    ".note{font-size:.85rem;color:#3a3f47;background:#f6f7f9;border-left:3px solid #c3c8d0;"
    "padding:10px 14px;margin:10px 0}"
    "ul{margin:8px 0;padding-left:20px}code{font-size:.85em;word-break:break-all}"
    "svg{width:100%;height:auto;background:#fafbfc;border:1px solid #d8dbe0;border-radius:6px}"
    ".legend{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:.8rem;color:#4a4f57;margin:4px 0}"
    ".legend .key{display:inline-flex;align-items:center;gap:5px}"
    ".legend i{width:14px;height:3px;border-radius:2px;display:inline-block}"
)


class ReportError(ValueError):
    """レポートを組み立てられない。"""


@dataclass(frozen=True, slots=True)
class ReportFigure:
    """レポートへ埋め込む図 1 枚。

    ``data_uri`` は `adapters.imaging.image_encoding.encode_data_uri` の出力。
    ``note`` は §5.3 が要求する但し書き（対称化画像なら「加工画像です」）で、
    無い図では空にする。
    """

    caption: str
    data_uri: str
    note: str = ""


def report_filename(bundle: object, *, mode: str) -> str:
    """§9.4 と同じ作法。**人物名も時刻も既定で含めない。**"""
    manifest = getattr(bundle, "manifest", None)
    session_id = str(getattr(manifest, "session_id", "") or "")
    short = session_id.replace("-", "")[:8] or "session"
    return f"face-symmetry-{mode}_{short}.html"


# --- 組み立ての小道具 ---------------------------------------------------------


def _text(value: object) -> str:
    """**外から来た文字列は必ずここを通す。**

    図の説明も母集団の但し書きも呼び出し側の文字列である。生のまま埋めると
    markup が壊れ、`<` を含む値で表示が崩れる。
    """
    return escape(str(value), quote=True)


def _rows(rows: Sequence[Sequence[str]]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return body


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_text(header)}</th>" for header in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{_rows(rows)}</tbody></table>"


def _paragraphs(lines: Sequence[str], *, css_class: str = "") -> str:
    attribute = f' class="{css_class}"' if css_class else ""
    return "".join(f"<p{attribute}>{_text(line)}</p>" for line in lines)


def _bullets(lines: Sequence[str]) -> str:
    items = "".join(f"<li>{_text(line)}</li>" for line in lines)
    return f"<ul>{items}</ul>"


def _figures_section(figures: Sequence[ReportFigure]) -> str:
    """AC-REPORT-03: 図が無ければ**節ごと出さない**。空の枠を残さない。"""
    if not figures:
        return ""
    blocks = []
    for figure in figures:
        note = f"<p class='meta'>{_text(figure.note)}</p>" if figure.note else ""
        blocks.append(
            "<figure>"
            f'<img src="{_text(figure.data_uri)}" alt="{_text(figure.caption)}">'
            f"<figcaption>{_text(figure.caption)}</figcaption>"
            f"{note}"
            "</figure>"
        )
    return "<h2>画像</h2>" + "".join(blocks)


def _notes_section(*, has_figures: bool) -> str:
    """§5.3.1: 注意はここ 1 か所・数行だけ。

    §5.3 の留保全文と §10.7 の禁止用途一覧を書き写さない。図が無いときは共有の
    1 行も出さない——起こらないことへの注意を出し続けると、注意そのものが
    読み流される。
    """
    notes = (*REPORT_NOTES, REPORT_SHARING_NOTE) if has_figures else REPORT_NOTES
    return "<div class='note'>" + _paragraphs(notes) + "</div>"


def _provenance_section(bundle: object) -> str:
    """FR-020 と同じ作法で、再現に必要な版一式を本体へ入れる。"""
    manifest = getattr(bundle, "manifest", None)
    if manifest is None:
        raise ReportError("レポートには manifest が必要です。")
    rows: list[list[str]] = [
        ["レポートの版", REPORT_VERSION],
        ["レコードスキーマ", str(getattr(manifest, "schema_version", "—"))],
        ["指標", str(getattr(manifest, "metric_version", "—"))],
        ["アプリ", str(getattr(manifest, "app_version", "—"))],
        ["依存ロック (sha256)", str(getattr(manifest, "lock_digest", "—"))],
    ]
    for model in getattr(manifest, "models", ()) or ():
        rows.append([f"モデル: {model.name}", f"{model.version} / {model.sha256}"])
    revisions = getattr(manifest, "config_revisions", ()) or ()
    if revisions:
        latest = revisions[-1]
        rows.append(["設定 fingerprint", str(latest.fingerprint)])
    return "<h2>この数値を再現するための情報</h2>" + _table(("項目", "値"), rows)


def _document(*, title: str, body: str) -> str:
    return (
        "<!doctype html>\n"
        '<html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_text(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>\n"
    )


def _header(title: str, subtitle: str) -> str:
    return f"<h1>{_text(title)}</h1><p class='meta'>{_text(subtitle)}</p>"


# --- Camera（§9.6.1） ---------------------------------------------------------


def _summary_table(section: SummarySection) -> str:
    rows: list[list[str]] = []
    for key, label in SUMMARY_ROW_LABELS:
        reading = reading_for(section, key)
        fmt = format_summary_points if key == "score" else format_summary_percent
        if reading is None:
            rows.append([label, "—", "—", "—", "—"])
            continue
        rows.append(
            [
                label,
                fmt(reading.median),
                fmt(reading.iqr) if reading.iqr is not None else "—",
                f"{fmt(reading.minimum)} 〜 {fmt(reading.maximum)}",
                f"{reading.sample_count:,}",
            ]
        )
    return _table(("値", "中央値", SUMMARY_SPREAD_LABEL, "最小 〜 最大", "件数"), rows)


def _section_block(section: SummarySection) -> str:
    span = "—" if section.valid_span_ms is None else f"{section.valid_span_ms / 1000.0:.1f} 秒"
    facts = _table(
        ("項目", "値"),
        [
            ["計測条件を満たしたフレーム", f"{section.valid_rate * 100.0:.1f}%"],
            ["記録した行数", f"{section.total_count:,}"],
            ["うち有効な行数", f"{section.ok_count:,}"],
            ["有効だった長さ", span],
        ],
    )
    reasons = ""
    if section.reason_counts:
        lines = [
            f"{REASON_GUIDANCE.get(code, code.value)}"
            f"（{count:,} 回 · {count / section.total_count if section.total_count else 0:.0%}）"
            for code, count in section.reason_counts
        ]
        reasons = "<h3>計測条件を満たさなかった主な理由</h3>" + _bullets(lines)
    return facts + _summary_table(section) + reasons


#: 推移図の系列色。HUD の折れ線（`ui.hud_assets`）と同じ割り当てにする——同じ
#: 部位が画面と保存物で違う色になると、並べて見たときに読み替えが要る。
_TREND_COLORS: Final[dict[str, str]] = {
    "overall": "#2b5f8f",
    "contour": "#4FC3F7",
    "eyes": "#7CB342",
    "nose": "#EF8B22",
    "mouth": "#D8467F",
}


def _polyline(values: Sequence[float], *, low: float, high: float, color: str) -> str:
    """1 系列を、窓 `low`〜`high` へ clamp して描く。"""
    if not values:
        return ""
    span = high - low
    step = _TREND_WIDTH_PX / max(1, len(values) - 1)
    points = " ".join(
        f"{index * step:.1f},"
        f"{_TREND_HEIGHT_PX - ((min(high, max(low, value)) - low) / span) * _TREND_HEIGHT_PX:.1f}"
        for index, value in enumerate(values)
    )
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" '
        'stroke-width="1.8" stroke-linejoin="round"></polyline>'
    )


def _chart(
    lines: Sequence[tuple[str, Sequence[float], str]],
    *,
    low: float,
    high: float,
    unit: str,
) -> str:
    """1 つの軸に載る系列をまとめて 1 枚の図にする。

    縦軸の窓は固定する。自動スケールにするとノイズが誇張されて「大きく動いた」と
    誤読される（`ui.hud` と同じ判断）。**空の系列では何も描かない**——軸だけの図は
    「測ったが平らだった」と読める。
    """
    drawn = [(label, values, color) for label, values, color in lines if values]
    if not drawn:
        return ""
    body = "".join(_polyline(values, low=low, high=high, color=color) for _, values, color in drawn)
    legend = "".join(
        f'<span class="key"><i style="background:{color}"></i>{_text(label)}</span>'
        for label, _, color in drawn
    )
    return (
        f'<svg viewBox="0 0 {_TREND_WIDTH_PX} {_TREND_HEIGHT_PX}" '
        'preserveAspectRatio="none" role="img">'
        f"{body}"
        f'<text x="6" y="14" font-size="11" fill="#4a4f57">{high:.0f}{_text(unit)}</text>'
        f'<text x="6" y="{_TREND_HEIGHT_PX - 6}" font-size="11" fill="#4a4f57">'
        f"{low:.0f}{_text(unit)}</text>"
        "</svg>"
        f'<div class="legend">{legend}</div>'
    )


def _trend_section(series: TrendSeries | None) -> str:
    """§9.6.1: 画面に出る 6 つすべての推移。

    **% と点で図を分ける。** 目盛りの違う 2 つを 1 つの軸へ重ねると、どちらの
    単位で読めばよいのか分からなくなる（§8.6.7 が両者を分けているのと同じ理由）。
    """
    if series is None or series.is_empty:
        return ""
    percent_chart = _chart(
        [
            ("全体の対称度", series.symmetry, _TREND_COLORS["overall"]),
            *(
                (PART_LABELS[name], series.parts.get(name, ()), _TREND_COLORS[name])
                for name in PART_NAMES
            ),
        ],
        low=_TREND_Y_MIN,
        high=_TREND_Y_MAX,
        unit="%",
    )
    score_chart = _chart(
        [(PART_EQUAL_WEIGHT_LABEL, series.score, _TREND_COLORS["overall"])],
        low=_SCORE_Y_MIN,
        high=_SCORE_Y_MAX,
        unit=" 点",
    )
    blocks = ""
    if percent_chart:
        blocks += "<h3>対称度の推移</h3>" + percent_chart
    if score_chart:
        blocks += f"<h3>{_text(PART_EQUAL_WEIGHT_LABEL)}の推移</h3>" + score_chart
    return blocks + "<p class='meta'>横軸は記録された順序です。縦軸の範囲外は端に貼り付きます。</p>"


def build_camera_report(
    *,
    summary: SessionSummary,
    bundle: object,
    series: TrendSeries | None = None,
) -> str:
    """§9.6.1: Camera の 1 セッション分。

    **全行の生データを持たない。** 3,000 行を HTML へ入れても読めず、台帳としては
    CSV / JSON が既にある。代わりにそちらを案内する。

    **顔写真も持たない**（2026-08-16 の指示）。リアルタイムのレポートで見たいのは
    1 セッションの推移であって 1 瞬間の顔ではない。持たせないことで、Camera 側は
    不変条件 1 の限定解除を必要としなくなる——顔写真がディスクへ出る経路は Image
    だけに閉じる。
    """
    if summary.overall is None:
        raise ReportError("記録が 1 件も無いセッションのレポートは作れません。")

    body = _header(
        "リアルタイム計測のまとめ",
        f"{REPORT_VERSION} · 時刻はセッション開始からの相対値です。",
    )
    body += "<h2>このセッションの要約</h2>" + _section_block(summary.overall)
    body += f"<p class='meta'>{_text(SUMMARY_LIMITS_NOTE)}</p>"
    body += _trend_section(series)

    if len(summary.segments) > 1:
        body += "<h2>区間ごとの内訳</h2>"
        body += f"<p class='meta'>{_text(SUMMARY_SEGMENT_NOTE)}</p>"
        for index, section in enumerate(summary.segments, start=1):
            short = (section.segment_id or "")[:8]
            label = f"区間 {index}（{short}）" if short else f"区間 {index}"
            body += f"<h3>{_text(label)}</h3>" + _section_block(section)

    body += "<p class='meta'>1 フレームごとの記録は CSV / JSON にあります。</p>"
    body += _notes_section(has_figures=False)
    body += _provenance_section(bundle)
    return _document(title="リアルタイム計測のまとめ", body=body)


# --- Image（§9.6.2） ----------------------------------------------------------


def _baseline_rows(
    baseline: BaselineSummary | None,
    *,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
) -> dict[str, str]:
    """FR-039: 併記があるときだけ「平均（差）」を作る。

    整形は `ui.readings` と共有する。レポートだけ別の丸め方をすると、画面の括弧の
    中と保存物の括弧の中が食い違う。
    """
    if baseline is None:
        return {}
    metrics = baseline.metrics
    rows = {
        "symmetry": format_baseline_symmetry(
            baseline_reading(metrics, OVERALL_METRIC, value=symmetry_percent(overall))
        ),
        "score": format_baseline_score(
            baseline_reading(metrics, SCORE_METRIC, value=score_points_value(equal_weight))
        ),
    }
    for name in PART_NAMES:
        rows[name] = format_baseline_symmetry(
            baseline_reading(metrics, PART_METRICS[name], value=symmetry_percent(parts.get(name)))
        )
    return rows


def _score_calculation(parts: Mapping[str, float | None]) -> str:
    """§8.6.3: 「全体のスコア」の求め方。画面と同じ `explain_score` を読む。

    画面の 3 ステップ表示（`ui.scores`）を字面まで写さないのは、あちらが等幅の
    整形表を Streamlit 前提で組んでいるためである。**値の出どころは同じ**なので、
    数字は画面と一致する。
    """
    explanation = explain_score(parts)
    if explanation is None:
        return (
            "<h3>「全体のスコア」の求め方</h3>"
            "<p class='meta'>部位が 1 つでも欠測しているため、この画像では計算できません。</p>"
        )
    rows = [
        [PART_LABELS.get(part.name, part.name), f"{part.distance * 100.0:.2f}%"]
        for part in explanation.parts
    ]
    rows.append(["4 部位の二乗平均平方根", f"{explanation.combined_distance * 100.0:.2f}%"])
    rows.append([PART_EQUAL_WEIGHT_LABEL, f"{explanation.points:.1f} 点"])
    anchors = (
        f"目盛りは 2 点で決まります。ずれ {explanation.distance_at_full_score * 100.0:.2f}% が"
        f" 100 点、ずれ {explanation.distance_at_zero_score * 100.0:.2f}% が 0 点です。"
        f"ずれが 1 ポイント動くと {explanation.points_per_deviation_percent:.0f} 点動きます。"
    )
    return (
        "<h3>「全体のスコア」の求め方</h3>"
        + _table(("部位", "ずれ"), rows)
        + f"<p class='meta'>{_text(anchors)}</p>"
    )


def build_image_report(
    *,
    outcome: AnalysisOutcome,
    bundle: object,
    figures: Sequence[ReportFigure] = (),
    baseline: BaselineSummary | None = None,
    estimate: DemographicEstimate | None = None,
    show_demographics: bool = False,
    notice: str = "",
) -> str:
    """§9.6.2: 写真 1 枚分の完全な記録。

    **要約は持たない。** 1 行しか無く、分布が存在しないためである（FR-021）。

    ``show_demographics`` が偽のときは参考年齢・参考性別を出さず、``notice``
    （`ui.age_gate_notice` の固定文言）があればそれだけを置く。母集団の年齢から
    年齢ゲートの内部判断を復元させないためである（§10.12-3）。同じ理由で、
    ``baseline`` も表示しているときにしか渡してはならない。
    """
    overall = resolve_overall_distance(outcome)
    equal_weight = resolve_equal_weight_distance(outcome)
    parts = {name: resolve_part_distance_3d(outcome, name) for name in PART_NAMES}
    annotations = _baseline_rows(
        baseline if show_demographics else None,
        overall=overall,
        equal_weight=equal_weight,
        parts=parts,
    )

    # FR-039: 併記が無いときは**列ごと落とす**。空の列が残ると「機能が動いていて
    # 失敗している」と読まれる（HUD が `:empty` で行を畳むのと同じ理由）。
    annotated = any(annotations.values())
    rows: list[list[str]] = [
        ["全体の対称度", format_symmetry(overall)],
        [PART_EQUAL_WEIGHT_LABEL, format_score(equal_weight)],
        *([PART_LABELS[name], format_symmetry(parts[name])] for name in PART_NAMES),
    ]
    headers: tuple[str, ...] = ("値", "測定")
    if annotated:
        headers = ("値", "測定", "参照統計の平均（差）")
        keys = ["symmetry", "score", *PART_NAMES]
        rows = [[*row, annotations.get(key, "")] for row, key in zip(rows, keys, strict=True)]

    body = _header(
        "写真 1 枚の計測結果",
        f"{REPORT_VERSION} · この結果は 1 枚の写真だけから作られています。",
    )
    body += "<h2>測定結果</h2>" + _table(headers, rows)
    if baseline is not None and show_demographics:
        body += (
            "<p class='meta'>"
            + _text(baseline_note(age=baseline.age, gender=baseline.gender, count=baseline.count))
            + "</p>"
        )
    body += _score_calculation(parts)

    pose = outcome.pose
    body += "<h3>撮影の状態</h3>" + _table(
        ("項目", "値"),
        [
            ["顔の向き yaw", "—" if pose is None else f"{pose.yaw_deg:.1f}°"],
            ["顔の向き pitch", "—" if pose is None else f"{pose.pitch_deg:.1f}°"],
            ["顔の向き roll", "—" if pose is None else f"{pose.roll_deg:.1f}°"],
            [
                "画面に占める顔の面積比",
                "—"
                if outcome.face_area_ratio is None
                else f"{outcome.face_area_ratio * 100.0:.1f}%",
            ],
            [
                "肌表面の参考値（実験値）",
                "—"
                if outcome.skin_quality is None or outcome.skin_quality.score is None
                else f"{outcome.skin_quality.score:.1f}",
            ],
        ],
    )

    if show_demographics and estimate is not None:
        gender = GENDER_LABELS.get(estimate.gender, "—")
        body += "<h3>参考年齢・参考性別</h3>" + _table(
            ("項目", "値"),
            [
                ["参考年齢", f"{estimate.estimated_age_years:.0f} 歳"],
                ["参考性別", gender],
            ],
        )
        body += (
            "<p class='meta'>モデルが推定した参考値です。断定された年齢・性別ではなく、"
            "左右差の結果とは無関係な別々の値です。</p>"
        )
    elif notice:
        body += f"<p class='note'>{_text(notice)}</p>"

    body += _figures_section(figures)
    body += _notes_section(has_figures=bool(figures))
    body += _provenance_section(bundle)
    return _document(title="写真 1 枚の計測結果", body=body)
