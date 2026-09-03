"""FR-042 / §9.6: 人が読む結果レポート（単一 HTML）。

CSV / JSON（§9.2〜§9.4）は再現のための台帳で、列は機械可読な識別子である。
こちらは読み物であり、両者は目的が違うので併存させる。台帳側のスキーマは
この module によって一切変わらない。

**この module は Streamlit を呼ばない。** 受け取った値から HTML 文字列を組み立て
て返すだけなので、`AppTest` を使わずに文字列として検証できる。ダウンロードの
提示は `ui.export_panel` が行う。

**設計の中心は「保存されたファイルは画面の文脈を持たない」ことである。**
画面では周りの caption・タブ・About が担っていた注意書きが、ファイル 1 つの中に
揃っていなければ、数字だけが独り歩きする。したがって §5.3 の留保・§10.7 の禁止
用途・再現に必要な版一式を、**必ず本体へ入れる**。

**外部リソースを参照しない**（AC-REPORT-02）。CSS は inline、図は `data:` URI。
開いた先で通信が起きるレポートは、ローカル完結という前提をファイル単体では
守れていない。

文言は既存の定数を再利用する（`ui.scores` の留保、`ui.readings` のフォーマッタ、
`ui.summary_panel` の要約の整形）。レポートだけが別の言い方・別の丸め方をすると、
画面と保存物で違うことを言うことになる。

**FR-043**: レポート本体も表示言語（`ui.i18n.t`）に従う（§9.6・AC-I18N-01）。
`t()` は同じモジュール内で解決できる literal しか受け取れないため（`tests/ui/
test_i18n.py`）、他 module が持つ定数（`ui.summary_panel.REFERENCE_AGE_RATE_LABEL`
など）はこの module から直接 `t()` へ渡せない。値は変えずに**同じ文言をこの
module 内の literal として複製し**、それを `t()` へ渡す
（`_OVERALL_SYMMETRY_LABEL` など）。部位名・性別だけは例外で、
`ui.readings.part_label` / `gender_label` という**共有の訳付きアクセサ**が既に
`t()` を内包しているため、この module は関数をそのまま呼ぶだけでよく、文言を
複製しない（以前はここでも `_part_label` / `_gender_label` として複製していたが、
2 か所が食い違う経路になるため寄せた）。foreign module の関数
（`ui.readings.format_symmetry`）が返す文字列は、実行時の関数呼び出し結果で
あって literal ではないため、この module からは訳せない——英語表示でも
日本語のまま残る（ADR-0010 Consequences の「既知の穴」と同じ扱い）。

**「全体のスコア」の求め方は撤回した**（2026-09-03、§0 項目 34）。§8.6.7 の
表示スケール（旧 `domain.score_explanation`）を機能ごと削除した。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Final

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.application.demographic_observations import DemographicOverview
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.session_summary import (
    SessionSummary,
    SummarySection,
    TrendSeries,
)
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import (
    format_gender_probabilities,
    format_symmetry,
    gender_label,
    part_label,
    resolve_overall_distance,
    resolve_part_distance_3d,
)
from face_symmetry_lab.ui.summary_panel import (
    SUMMARY_LIMITS_NOTE,
    SUMMARY_ROW_LABELS,
    SUMMARY_SEGMENT_NOTE,
    format_summary_percent,
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
_CAPTURE_STATE_HEADING: Final[str] = "撮影の状態"
_YAW_LABEL: Final[str] = "顔の向き yaw"
_PITCH_LABEL: Final[str] = "顔の向き pitch"
_ROLL_LABEL: Final[str] = "顔の向き roll"
_FACE_AREA_LABEL: Final[str] = "画面に占める顔の面積比"

_VALUE_HEADER: Final[str] = "値"
_MEDIAN_HEADER: Final[str] = "中央値"
_RANGE_HEADER: Final[str] = "最小 〜 最大"
_COUNT_HEADER: Final[str] = "件数"
_ITEM_HEADER: Final[str] = "項目"
_MEASURED_HEADER: Final[str] = "測定"

#: `ui.summary_panel.SUMMARY_SPREAD_LABEL` と同じ文言。他 module の import なので
#: `t()` へ直接渡せず、ここでも literal として複製する（上の FR-043 の節と同じ理由）。
_SPREAD_LABEL: Final[str] = "変動幅（真ん中50%）"

_VALID_RATE_LABEL: Final[str] = "計測条件を満たしたフレーム"
_RECORDED_ROWS_LABEL: Final[str] = "記録した行数"
_VALID_ROWS_LABEL: Final[str] = "うち有効な行数"
_VALID_SPAN_LABEL: Final[str] = "有効だった長さ"
_REFERENCE_AGE_RATE_LABEL: Final[str] = "参考年齢を表示できたフレーム"
_SPAN_SECONDS_TEMPLATE: Final[str] = "{value:.1f} 秒"
_REASON_LINE_TEMPLATE: Final[str] = "{guidance}（{count:,} 回 · {share:.0%}）"

_SYMMETRY_TREND_HEADING: Final[str] = "対称度の推移"
_TREND_AXIS_NOTE: Final[str] = "横軸は記録された順序です。縦軸の範囲外は端に貼り付きます。"

_AGE_SINGLE_TEMPLATE: Final[str] = "{median} 歳"
_AGE_RANGE_TEMPLATE: Final[str] = "{median} 歳（真ん中50%: {low}〜{high} 歳）"
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


def _summary_table(section: SummarySection) -> str:
    """要約の表。中央値・変動幅（真ん中50%）・最小〜最大・件数を部位ごとに並べる。"""
    rows: list[list[str]] = []
    for key, _label in SUMMARY_ROW_LABELS:
        label = summary_row_label(key)
        reading = reading_for(section, key)
        if reading is None:
            row = [label, "—", "—", "—", "—"]
        else:
            row = [
                label,
                format_summary_percent(reading.median),
                format_summary_percent(reading.iqr) if reading.iqr is not None else "—",
                f"{format_summary_percent(reading.minimum)} 〜 "
                f"{format_summary_percent(reading.maximum)}",
                f"{reading.sample_count:,}",
            ]
        rows.append(row)
    headers = [
        t(_VALUE_HEADER),
        t(_MEDIAN_HEADER),
        t(_SPREAD_LABEL),
        t(_RANGE_HEADER),
        t(_COUNT_HEADER),
    ]
    return _table(headers, rows)


def _section_block(section: SummarySection) -> str:
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
    return facts_table + _summary_table(section) + reasons


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
    """§9.6.1: 画面に出る 5 つすべての推移。

    **5 系列とも同じ % なので 1 つの図へ重ねる**（§0 項目 34 で「全体のスコア」
    の点の目盛りを撤回したため）。
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
    if not percent_chart:
        return ""
    blocks = f"<h3>{_text(t(_SYMMETRY_TREND_HEADING))}</h3>" + percent_chart
    return blocks + f"<p class='meta'>{_text(t(_TREND_AXIS_NOTE))}</p>"


def _session_demographic_section(
    overview: DemographicOverview | None,
    *,
    notice: str = "",
) -> str:
    """§9.6.1（2026-08-26）: Image と同じ形で、そのセッションの参考年齢・参考性別。

    リアルタイムは 1 点を持たないため、**実際に表示できた判定**の中央値と真ん中
    50% の範囲、観測フレームのモデル性別確率の平均、件数を出す。丸めと範囲は
    レポートの参考年齢要約が使う `domain.demographics.observed_ages` から来る
    （§9.6.1）。

    範囲が 1 歳に収まったセッションでは「31〜31 歳」と書かない——同じ数を 2 度
    並べても幅は伝わらない。
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
    記録から作った列を渡してはならない（§10.11-2）。1 件も表示できなかった
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
    body += _section_block(summary.overall)
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


def build_image_report(
    *,
    outcome: AnalysisOutcome,
    bundle: object,
    figures: Sequence[ReportFigure] = (),
    estimate: DemographicEstimate | None = None,
    show_demographics: bool = False,
    notice: str = "",
) -> str:
    """§9.6.2: 写真 1 枚分の完全な記録。

    **要約は持たない。** 1 行しか無く、分布が存在しないためである（FR-021）。

    ``show_demographics`` が偽のときは参考年齢・参考性別を出さず、``notice``
    （`ui.age_gate_notice` の固定文言）があればそれだけを置く。
    """
    overall = resolve_overall_distance(outcome)
    parts = {name: resolve_part_distance_3d(outcome, name) for name in PART_NAMES}

    rows: list[list[str]] = [
        [t(_OVERALL_SYMMETRY_LABEL), format_symmetry(overall)],
        *([part_label(name), format_symmetry(parts[name])] for name in PART_NAMES),
    ]
    headers: tuple[str, ...] = (t(_VALUE_HEADER), t(_MEASURED_HEADER))

    body = _header(
        t(_IMAGE_REPORT_TITLE),
        t(_IMAGE_SUBTITLE_TEMPLATE).format(version=REPORT_VERSION),
    )
    body += f"<h2>{_text(t(_MEASUREMENT_HEADING))}</h2>" + _table(headers, rows)

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
