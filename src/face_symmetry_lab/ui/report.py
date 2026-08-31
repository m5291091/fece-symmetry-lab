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

**FR-043**: レポート本体も表示言語（`ui.i18n.t`）に従う（§9.6・AC-I18N-01）。
`t()` は同じモジュール内で解決できる literal しか受け取れないため（`tests/ui/
test_i18n.py`）、他 module が持つ定数（`ui.scores.PART_EQUAL_WEIGHT_LABEL`、
`ui.summary_panel.REFERENCE_AGE_RATE_LABEL` など）はこの module から直接
`t()` へ渡せない。値は変えずに**同じ文言をこの module 内の literal として複製し**、
それを `t()` へ渡す（`_OVERALL_SCORE_LABEL` など）。部位名・性別だけは例外で、
`ui.readings.part_label` / `gender_label` という**共有の訳付きアクセサ**が既に
`t()` を内包しているため、この module は関数をそのまま呼ぶだけでよく、文言を
複製しない（以前はここでも `_part_label` / `_gender_label` として複製していたが、
2 か所が食い違う経路になるため寄せた）。foreign module の関数
（`ui.readings.format_symmetry` / `format_score` / `baseline_note` /
`format_baseline_symmetry` / `format_baseline_score`）が返す文字列は、実行時の
関数呼び出し結果であって literal ではないため、この module からは訳せない
——英語表示でも日本語のまま残る（ADR-0010 Consequences の「既知の穴」と同じ扱い）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Final

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.demographic_observations import DemographicOverview
from face_symmetry_lab.application.reference_baseline import (
    BaselineSummary,
    SessionBaselineSummary,
)
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
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import (
    baseline_note,
    format_baseline_score,
    format_baseline_symmetry,
    format_gender_probabilities,
    format_score,
    format_symmetry,
    gender_label,
    part_label,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
    score_points_value,
    symmetry_percent,
)
from face_symmetry_lab.ui.summary_panel import (
    SUMMARY_LIMITS_NOTE,
    SUMMARY_ROW_LABELS,
    SUMMARY_SEGMENT_NOTE,
    format_summary_percent,
    format_summary_points,
    reading_for,
    summary_row_label,
)

#: §9.6: レポートそのものの版。レコードスキーマ（`4.0.0`）とは別に数える——
#: 台帳の列が変わらなくても、レポートの構成は変わりうる。
REPORT_VERSION: Final[str] = "face-symmetry-report/1.0.0"

#: §10.4 / ADR-0009 (d): 図を入れたときだけ足す 1 行。Camera のレポートは図を
#: 持たないので、この文も出ない。
REPORT_SHARING_NOTE: Final[str] = (
    "このファイルには顔写真が含まれます。保存後の共有先はアプリからは制御できません。"
)

#: §9.6（2026-08-26）: **モード共通の節見出し。** Camera と Image が別々の文字列を
#: 持つと、同じセッションを画面で見た人とファイルで受け取った人が別の目次を読む。
DEMOGRAPHIC_HEADING: Final[str] = "参考年齢・参考性別"

#: §5.3 / 不変条件 5: 数値と同じ場所に置く 1 行。**左右差との関係を語らない。**
DEMOGRAPHIC_NOTE: Final[str] = (
    "モデルが推定した参考値です。断定された年齢・性別ではなく、左右差の結果とは無関係な別々の値です。"
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
    # 参考年齢・参考性別を、両モードで同じ形に見せるための最小の枠（§9.6）。
    # **色で意味を作らない**（§5.2）——枠と文字の大きさだけで、読む順序を示す。
    ".tiles{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}"
    ".tile{flex:1 1 200px;border:1px solid #d8dbe0;border-radius:8px;padding:10px 14px;"
    "background:#fafbfc}"
    ".tile .k{display:block;font-size:.8rem;color:#4a4f57}"
    ".tile .v{display:block;font-size:1.5rem;font-weight:600;margin-top:2px;"
    "font-variant-numeric:tabular-nums}"
)

# --- FR-043: 他 module の定数を、この module 内で解決できる形へ複製する -------
#
# `t()` の引数は「文字列リテラル」か「同じモジュールの module-level で文字列
# リテラルへ束縛された名前」でなければならない（`tests/ui/test_i18n.py`）。
# ここに残る定数はどちらにも当たらない他 module の値なので、**値は変えずに
# 同じ文言をここでも literal として持つ**。部位名・性別は `ui.readings.part_label`
# / `gender_label` が既に訳付きアクセサとして共有しているため、ここでは複製しない
# （上の import を参照）。

_OVERALL_SYMMETRY_LABEL: Final[str] = "全体の対称度"
_OVERALL_SCORE_LABEL: Final[str] = "全体のスコア"


_CAMERA_REPORT_TITLE: Final[str] = "リアルタイム計測のまとめ"
_IMAGE_REPORT_TITLE: Final[str] = "写真 1 枚の計測結果"
_CAMERA_SUBTITLE_TEMPLATE: Final[str] = "{version} · 時刻はセッション開始からの相対値です。"
_IMAGE_SUBTITLE_TEMPLATE: Final[str] = "{version} · この結果は 1 枚の写真だけから作られています。"

_MEASUREMENT_HEADING: Final[str] = "測定結果"
_SESSION_SUMMARY_HEADING: Final[str] = "このセッションの要約"
_SEGMENT_BREAKDOWN_HEADING: Final[str] = "区間ごとの内訳"
_SEGMENT_LABEL_TEMPLATE: Final[str] = "区間 {index}"
_SEGMENT_LABEL_WITH_ID_TEMPLATE: Final[str] = "区間 {index}（{short}）"
_HOLD_REASONS_HEADING: Final[str] = "計測条件を満たさなかった主な理由"
_IMAGES_HEADING: Final[str] = "画像"
_SCORE_CALCULATION_HEADING: Final[str] = "「全体のスコア」の求め方"
_SCORE_CALCULATION_UNAVAILABLE_NOTE: Final[str] = (
    "部位が 1 つでも欠測しているため、この画像では計算できません。"
)
_COMBINED_RMS_LABEL: Final[str] = "4 部位の二乗平均平方根"
_SCALE_ANCHORS_TEMPLATE: Final[str] = (
    "目盛りは 2 点で決まります。ずれ {full_score_pct:.2f}% が"
    " 100 点、ずれ {zero_score_pct:.2f}% が 0 点です。"
    "ずれが 1 ポイント動くと {points_per_deviation:.0f} 点動きます。"
)
_CAPTURE_STATE_HEADING: Final[str] = "撮影の状態"
_YAW_LABEL: Final[str] = "顔の向き yaw"
_PITCH_LABEL: Final[str] = "顔の向き pitch"
_ROLL_LABEL: Final[str] = "顔の向き roll"
_FACE_AREA_LABEL: Final[str] = "画面に占める顔の面積比"

_VALUE_HEADER: Final[str] = "値"
_MEDIAN_HEADER: Final[str] = "中央値"
_RANGE_HEADER: Final[str] = "最小 〜 最大"
_COUNT_HEADER: Final[str] = "件数"
_BASELINE_MEAN_HEADER: Final[str] = "参照統計の平均"
_ITEM_HEADER: Final[str] = "項目"
_MEASURED_HEADER: Final[str] = "測定"
_BASELINE_MEAN_DIFF_HEADER: Final[str] = "参照統計の平均（差）"
_PART_HEADER: Final[str] = "部位"
_DEVIATION_HEADER: Final[str] = "ずれ"

#: `ui.summary_panel.SUMMARY_SPREAD_LABEL` と同じ文言。他 module の import なので
#: `t()` へ直接渡せず、ここでも literal として複製する（上の FR-043 の節と同じ理由）。
_SPREAD_LABEL: Final[str] = "変動幅（真ん中50%）"

_VALID_RATE_LABEL: Final[str] = "計測条件を満たしたフレーム"
_RECORDED_ROWS_LABEL: Final[str] = "記録した行数"
_VALID_ROWS_LABEL: Final[str] = "うち有効な行数"
_VALID_SPAN_LABEL: Final[str] = "有効だった長さ"
_REFERENCE_AGE_RATE_LABEL: Final[str] = "参考年齢を表示できたフレーム"
_SPAN_SECONDS_TEMPLATE: Final[str] = "{value:.1f} 秒"
_POINTS_TEMPLATE: Final[str] = "{value:.1f} 点"
_REASON_LINE_TEMPLATE: Final[str] = "{guidance}（{count:,} 回 · {share:.0%}）"

_SYMMETRY_TREND_HEADING: Final[str] = "対称度の推移"
_SCORE_TREND_HEADING: Final[str] = "全体のスコアの推移"
_TREND_AXIS_NOTE: Final[str] = "横軸は記録された順序です。縦軸の範囲外は端に貼り付きます。"
_SCORE_UNIT_SUFFIX: Final[str] = " 点"

_AGE_SINGLE_TEMPLATE: Final[str] = "{median} 歳"
_AGE_RANGE_TEMPLATE: Final[str] = "{median} 歳（真ん中50%: {low}〜{high} 歳）"
_POPULATION_LINE_TEMPLATE: Final[str] = (
    "平均: 参照統計の {span}・{genders}（人種で絞り込まず {count:,} 件）。探索値です。"
)
_SHOWN_COUNT_TEMPLATE: Final[str] = "表示できた判定 {count:,} 件"
_SHOWN_RANGE_TEMPLATE: Final[str] = "真ん中50%: {low}〜{high} 歳 · {meta}"
_AGE_YEARS_TEMPLATE: Final[str] = "{value:.0f} 歳"
_MEDIAN_AGE_TILE_LABEL: Final[str] = "参考年齢（中央値）"
_REFERENCE_AGE_TILE_LABEL: Final[str] = "参考年齢"
_REFERENCE_GENDER_TILE_LABEL: Final[str] = "参考性別"


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


def _tiles(items: Sequence[tuple[str, str]]) -> str:
    """ラベルと値の組を、そのまま読める大きさで並べる。

    表にしないのは、2 項目しか無いものを行と列で組むと本文より小さく見えるため
    である。§5.2 のとおり色・矢印・進捗の演出は持たせない。
    """
    cells = "".join(
        f'<div class="tile"><span class="k">{_text(label)}</span>'
        f'<span class="v">{_text(value)}</span></div>'
        for label, value in items
    )
    return f'<div class="tiles">{cells}</div>'


def _demographic_section(
    items: Sequence[tuple[str, str]],
    *,
    meta: str = "",
    notice: str = "",
) -> str:
    """§9.6（2026-08-26）: 参考年齢・参考性別を、Camera と Image で同じ形に出す。

    数値が無いときは節ごと出さない（`items` が空）。ただし ``notice``——§10.11-2 の
    固定文言——があるときは、その 1 行だけを置く。**空の見出しを残さない**のは、
    図の節（`_figures_section`）と同じ理由である。
    """
    if not items:
        return f"<p class='note'>{_text(notice)}</p>" if notice else ""
    block = f"<h2>{_text(t(DEMOGRAPHIC_HEADING))}</h2>" + _tiles(items)
    if meta:
        block += f"<p class='meta'>{_text(meta)}</p>"
    return block + f"<p class='meta'>{_text(t(DEMOGRAPHIC_NOTE))}</p>"


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
    return f"<h2>{_text(t(_IMAGES_HEADING))}</h2>" + "".join(blocks)


def _sharing_section(*, has_figures: bool) -> str:
    """§10.4 / ADR-0009 (d): 図を入れたときだけ出す 1 行。

    2026-08-16 の指示で、レポートの注意文の節と来歴の表は削除した。**残すのはここ
    だけである**——顔写真がディスクへ出る唯一の経路であり、受け取った人が画面を
    見ずに読む可能性がある。図が無いときは何も出さない（Camera は常にこちら）。

    FR-020 の再現性は CSV / JSON が担う。あちらは各行が版一式を自己完結して持って
    おり（§9.4）、同じ情報を 2 つの成果物へ書き写す必要はない。
    """
    if not has_figures:
        return ""
    return f"<p class='note'>{_text(t(REPORT_SHARING_NOTE))}</p>"


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


def _summary_table(section: SummarySection, *, baseline: SessionBaselineSummary | None) -> str:
    """要約の表。母集団が決まっているときだけ「参照統計の平均」列を足す。

    FR-039 の併記と違い**差の列は作らない**——中央値と平均は別の母集団の代表値
    どうしなので、引き算しても「表示している 1 つの数との差」にはならない。
    """
    means = _session_baseline_rows(baseline)
    rows: list[list[str]] = []
    for key, _label in SUMMARY_ROW_LABELS:
        label = summary_row_label(key)
        reading = reading_for(section, key)
        fmt = format_summary_points if key == "score" else format_summary_percent
        if reading is None:
            row = [label, "—", "—", "—", "—"]
        else:
            row = [
                label,
                fmt(reading.median),
                fmt(reading.iqr) if reading.iqr is not None else "—",
                f"{fmt(reading.minimum)} 〜 {fmt(reading.maximum)}",
                f"{reading.sample_count:,}",
            ]
        if means:
            row.append(means.get(key, "—"))
        rows.append(row)
    headers = [
        t(_VALUE_HEADER),
        t(_MEDIAN_HEADER),
        t(_SPREAD_LABEL),
        t(_RANGE_HEADER),
        t(_COUNT_HEADER),
    ]
    if means:
        headers.append(t(_BASELINE_MEAN_HEADER))
    return _table(headers, rows)


def _section_block(
    section: SummarySection,
    *,
    baseline: SessionBaselineSummary | None = None,
) -> str:
    span = (
        "—"
        if section.valid_span_ms is None
        else t(_SPAN_SECONDS_TEMPLATE).format(value=section.valid_span_ms / 1000.0)
    )
    facts: list[list[str]] = [
        [t(_VALID_RATE_LABEL), f"{section.valid_rate * 100.0:.1f}%"],
        [t(_RECORDED_ROWS_LABEL), f"{section.total_count:,}"],
        [t(_VALID_ROWS_LABEL), f"{section.ok_count:,}"],
        [t(_VALID_SPAN_LABEL), span],
    ]
    # FR-021 (5): ゲートを呼んだセッションでだけ行を足す。§10.11-2 のとおり、
    # できなかった内訳は持たない。
    rate = section.reference_age_rate
    if rate is not None:
        facts.append([t(_REFERENCE_AGE_RATE_LABEL), f"{rate * 100.0:.0f}%"])
    facts_table = _table((t(_ITEM_HEADER), t(_VALUE_HEADER)), facts)
    reasons = ""
    if section.reason_counts:
        lines = [
            t(_REASON_LINE_TEMPLATE).format(
                guidance=t(REASON_GUIDANCE[code]),
                count=count,
                share=count / section.total_count if section.total_count else 0,
            )
            for code, count in section.reason_counts
        ]
        reasons = f"<h3>{_text(t(_HOLD_REASONS_HEADING))}</h3>" + _bullets(lines)
    return facts_table + _summary_table(section, baseline=baseline) + reasons


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
            (t(_OVERALL_SYMMETRY_LABEL), series.symmetry, _TREND_COLORS["overall"]),
            *(
                (part_label(name), series.parts.get(name, ()), _TREND_COLORS[name])
                for name in PART_NAMES
            ),
        ],
        low=_TREND_Y_MIN,
        high=_TREND_Y_MAX,
        unit="%",
    )
    score_chart = _chart(
        [(t(_OVERALL_SCORE_LABEL), series.score, _TREND_COLORS["overall"])],
        low=_SCORE_Y_MIN,
        high=_SCORE_Y_MAX,
        unit=t(_SCORE_UNIT_SUFFIX),
    )
    blocks = ""
    if percent_chart:
        blocks += f"<h3>{_text(t(_SYMMETRY_TREND_HEADING))}</h3>" + percent_chart
    if score_chart:
        blocks += f"<h3>{_text(t(_SCORE_TREND_HEADING))}</h3>" + score_chart
    return blocks + f"<p class='meta'>{_text(t(_TREND_AXIS_NOTE))}</p>"


def _session_baseline_rows(baseline: SessionBaselineSummary | None) -> dict[str, str]:
    """FR-039/§9.6.1: セッションの母集団に対する「平均」の表示文字列。

    Image の併記（`_baseline_rows`）と違い、**差は出さない。** 中央値と平均を
    引き算しても、表示している 1 つの数どうしの差ではなくなる（中央値は分布の
    代表値、平均は別の母集団の代表値である）。並べるのは平均だけにする。
    """
    if baseline is None:
        return {}
    metrics = baseline.metrics
    rows = {
        "symmetry": _baseline_mean(metrics, OVERALL_METRIC, points=False),
        "score": _baseline_mean(metrics, SCORE_METRIC, points=True),
    }
    for name in PART_NAMES:
        rows[name] = _baseline_mean(metrics, PART_METRICS[name], points=False)
    return rows


def _baseline_mean(metrics: Mapping[str, object], name: str, *, points: bool) -> str:
    summary = metrics.get(name)
    mean = getattr(summary, "mean", None)
    if mean is None:
        return "—"
    return t(_POINTS_TEMPLATE).format(value=mean) if points else f"{mean:.2f}%"


def _population_line(baseline: SessionBaselineSummary | None) -> str:
    """§10.12-5: 母集団が何であるかを、平均と同じ場所へ 1 行で書く。"""
    if baseline is None:
        return ""
    genders = "・".join(gender_label(gender) for gender in baseline.genders)
    ages = baseline.ages
    span = (
        t(_AGE_SINGLE_TEMPLATE).format(median=ages.median)
        if ages.low == ages.high
        else t(_AGE_RANGE_TEMPLATE).format(median=ages.median, low=ages.low, high=ages.high)
    )
    return t(_POPULATION_LINE_TEMPLATE).format(span=span, genders=genders, count=baseline.count)


def _session_demographic_section(
    overview: DemographicOverview | None,
    *,
    notice: str = "",
) -> str:
    """§9.6.1（2026-08-26）: Image と同じ形で、そのセッションの参考年齢・参考性別。

    リアルタイムは 1 点を持たないため、**実際に表示できた判定**の中央値と真ん中
    50% の範囲、観測フレームのモデル性別確率の平均、件数を出す。丸めと範囲は
    参照平均の母集団と同じ `observed_ages` から来る（§10.12-2）。

    範囲が 1 歳に収まったセッションでは「31〜31 歳」と書かない——同じ数を 2 度
    並べても幅は伝わらない（`_population_line` と同じ作法）。
    """
    if overview is None:
        return _demographic_section((), notice=notice)
    ages = overview.ages
    genders = format_gender_probabilities(overview.gender_probabilities)
    meta = t(_SHOWN_COUNT_TEMPLATE).format(count=ages.sample_count)
    if ages.low != ages.high:
        meta = t(_SHOWN_RANGE_TEMPLATE).format(low=ages.low, high=ages.high, meta=meta)
    return _demographic_section(
        (
            (t(_MEDIAN_AGE_TILE_LABEL), t(_AGE_SINGLE_TEMPLATE).format(median=ages.median)),
            (t(_REFERENCE_GENDER_TILE_LABEL), genders),
        ),
        meta=meta,
    )


def build_camera_report(
    *,
    summary: SessionSummary,
    bundle: object,
    series: TrendSeries | None = None,
    baseline: SessionBaselineSummary | None = None,
    overview: DemographicOverview | None = None,
    notice: str = "",
) -> str:
    """§9.6.1: Camera の 1 セッション分。

    **全行の生データを持たない。** 3,000 行を HTML へ入れても読めず、台帳としては
    CSV / JSON が既にある。代わりにそちらを案内する。

    **顔写真も持たない**（2026-08-16 の指示）。リアルタイムのレポートで見たいのは
    1 セッションの推移であって 1 瞬間の顔ではない。持たせないことで、Camera 側は
    不変条件 1 の限定解除を必要としなくなる——顔写真がディスクへ出る経路は Image
    だけに閉じる。

    ``overview``（2026-08-26）は、そのセッションで**実際に表示できた**参考年齢・
    参考性別である。渡してよいのは
    `application.demographic_observations.summarize_observations` の結果だけで、
    記録から作った列を渡してはならない（§10.12-3）。1 件も表示できなかった
    セッションでは節ごと出さず、``notice`` があればその 1 行だけを置く。

    節の並び（§9.6）は Image と同じである——見出し → 測定結果 → 参考年齢・参考性別。
    """
    if summary.overall is None:
        raise ReportError("記録が 1 件も無いセッションのレポートは作れません。")

    body = _header(
        t(_CAMERA_REPORT_TITLE),
        t(_CAMERA_SUBTITLE_TEMPLATE).format(version=REPORT_VERSION),
    )
    body += (
        f"<h2>{_text(t(_MEASUREMENT_HEADING))}</h2><h3>{_text(t(_SESSION_SUMMARY_HEADING))}</h3>"
    )
    body += _section_block(summary.overall, baseline=baseline)
    if baseline is not None:
        body += f"<p class='meta'>{_text(_population_line(baseline))}</p>"
    body += f"<p class='meta'>{_text(t(SUMMARY_LIMITS_NOTE))}</p>"
    body += _trend_section(series)

    if len(summary.segments) > 1:
        body += f"<h3>{_text(t(_SEGMENT_BREAKDOWN_HEADING))}</h3>"
        body += f"<p class='meta'>{_text(t(SUMMARY_SEGMENT_NOTE))}</p>"
        for index, section in enumerate(summary.segments, start=1):
            # 区間の見出しは太字の段落にする。`<h3>` にすると「区間ごとの内訳」と
            # 同じ深さになり、どちらが親なのか読めなくなる。画面（`ui.summary_panel`）
            # も同じく `**区間 N**` で出しているので、見出しの深さが両者で揃う。
            short = (section.segment_id or "")[:8]
            label = (
                t(_SEGMENT_LABEL_WITH_ID_TEMPLATE).format(index=index, short=short)
                if short
                else t(_SEGMENT_LABEL_TEMPLATE).format(index=index)
            )
            body += f"<p><strong>{_text(label)}</strong></p>" + _section_block(section)

    body += _session_demographic_section(overview, notice=notice)

    return _document(title=t(_CAMERA_REPORT_TITLE), body=body)


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
            f"<h3>{_text(t(_SCORE_CALCULATION_HEADING))}</h3>"
            f"<p class='meta'>{_text(t(_SCORE_CALCULATION_UNAVAILABLE_NOTE))}</p>"
        )
    rows = [[part_label(part.name), f"{part.distance * 100.0:.2f}%"] for part in explanation.parts]
    rows.append([t(_COMBINED_RMS_LABEL), f"{explanation.combined_distance * 100.0:.2f}%"])
    rows.append([t(_OVERALL_SCORE_LABEL), t(_POINTS_TEMPLATE).format(value=explanation.points)])
    anchors = t(_SCALE_ANCHORS_TEMPLATE).format(
        full_score_pct=explanation.distance_at_full_score * 100.0,
        zero_score_pct=explanation.distance_at_zero_score * 100.0,
        points_per_deviation=explanation.points_per_deviation_percent,
    )
    return (
        f"<h3>{_text(t(_SCORE_CALCULATION_HEADING))}</h3>"
        + _table((t(_PART_HEADER), t(_DEVIATION_HEADER)), rows)
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
        [t(_OVERALL_SYMMETRY_LABEL), format_symmetry(overall)],
        [t(_OVERALL_SCORE_LABEL), format_score(equal_weight)],
        *([part_label(name), format_symmetry(parts[name])] for name in PART_NAMES),
    ]
    headers: tuple[str, ...] = (t(_VALUE_HEADER), t(_MEASURED_HEADER))
    if annotated:
        headers = (t(_VALUE_HEADER), t(_MEASURED_HEADER), t(_BASELINE_MEAN_DIFF_HEADER))
        keys = ["symmetry", "score", *PART_NAMES]
        rows = [[*row, annotations.get(key, "")] for row, key in zip(rows, keys, strict=True)]

    body = _header(
        t(_IMAGE_REPORT_TITLE),
        t(_IMAGE_SUBTITLE_TEMPLATE).format(version=REPORT_VERSION),
    )
    body += f"<h2>{_text(t(_MEASUREMENT_HEADING))}</h2>" + _table(headers, rows)
    if baseline is not None and show_demographics:
        body += (
            "<p class='meta'>"
            + _text(baseline_note(age=baseline.age, gender=baseline.gender, count=baseline.count))
            + "</p>"
        )
    body += _score_calculation(parts)

    pose = outcome.pose
    body += f"<h3>{_text(t(_CAPTURE_STATE_HEADING))}</h3>" + _table(
        (t(_ITEM_HEADER), t(_VALUE_HEADER)),
        [
            [t(_YAW_LABEL), "—" if pose is None else f"{pose.yaw_deg:.1f}°"],
            [t(_PITCH_LABEL), "—" if pose is None else f"{pose.pitch_deg:.1f}°"],
            [t(_ROLL_LABEL), "—" if pose is None else f"{pose.roll_deg:.1f}°"],
            [
                t(_FACE_AREA_LABEL),
                "—"
                if outcome.face_area_ratio is None
                else f"{outcome.face_area_ratio * 100.0:.1f}%",
            ],
        ],
    )

    # §9.6（2026-08-26）: 見出し・並び・整形は Camera と同じ組み立てを通す。
    shown = estimate if show_demographics else None
    body += _demographic_section(
        ()
        if shown is None
        else (
            (
                t(_REFERENCE_AGE_TILE_LABEL),
                t(_AGE_YEARS_TEMPLATE).format(value=shown.estimated_age_years),
            ),
            (t(_REFERENCE_GENDER_TILE_LABEL), gender_label(shown.gender)),
        ),
        notice=notice,
    )

    body += _figures_section(figures)
    body += _sharing_section(has_figures=bool(figures))
    return _document(title=t(_IMAGE_REPORT_TITLE), body=body)
