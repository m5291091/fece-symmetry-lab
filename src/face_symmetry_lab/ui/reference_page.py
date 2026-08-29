"""FR-037 / §9.5: 参照統計ブラウザ（人種・性別・年齢で絞り込む記述統計）。

`configs/reference/` に同梱した事前集計セルテーブルを、絞り込みと集計だけで
表示する。**この module は推論式・幾何計算・ファイル永続化を持たない。** 絞り込み
と集計は `application.reference_browser` / `domain.reference_stats` を呼ぶだけで
ある。

この画面が扱うのは UTKFace の配布時注釈（人種・性別・年齢）に対する記述統計で
あり、本アプリはこれらを推定しない（不変条件 7）。**この画面は利用者自身の測定結果を
一切持ち込まない**（§10.7）——偏差値・順位・パーセンタイル・「平均より上」「平均より下」を
作らず、測定結果と参照統計をこの画面へ並べない。

**FR-039（吹き出し・HUD への参照平均の併記）とは別の画面である。** 2026-08-16 に
§10.7 を限定解除したのは Analyze 側の併記だけで（§10.12、ADR-0006）、この画面の
禁止は解除していない。同じセルテーブルを読むが、こちらは人種・性別・年齢を利用者が
自由に絞り込んで母集団そのものを眺めるためのページであり、個人の測定値は現れない。

**`metadata.warning` は `st.text` で表示する。** `st.markdown`/`st.caption` は
Markdown として解釈するため、原文に記号が含まれると意図しない見た目になりうる。
`st.text` は書式解釈をしない、最も「そのまま」に近い widget である。

**FR-043**: `metadata.warning`（配布元注意書きの原文引用）、`AGE_BANDS` /
`RACE_LABELS` / `GENDER_LABELS`（`domain.reference_stats` のデータ列の値）、
`dataset_name` / `license_scope` / `dataset_url` / 各 version / sha256 は
訳さない。前者は原文一致を要求する引用であり、後二者はデータそのものである。
`_AGE_MODE_BAND` / `_AGE_MODE_RANGE` は `st.radio` の内部値（比較にも使う）を
維持したまま、`format_func` で表示だけを訳す（ADR-0010 (c)、`ui.camera` の
`_BACKGROUND_MODE_LABELS` と同じ作法）。**`resolved.unavailable_reason` の
中身（`ReferenceDatasetError` の文言）が値を埋め込んで組み立てられている場合は
原文一致せず訳せない**——それ自体は `ui.i18n.translate_message`（UI 境界専用）
で訳を試みるが、静的でなければ日本語のまま残る（既知の穴、ADR-0010
Consequences）。外枠の文（`_UNAVAILABLE_WARNING_TEMPLATE`）は静的な template
なので `t()` で訳す。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.adapters.io.reference_dataset import CsvReferenceCellRepository
from face_symmetry_lab.application.reference_browser import (
    ReferenceBrowser,
    ReferenceQueryResult,
)
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.reference_stats import (
    AGE_BANDS,
    GENDER_LABELS,
    MAX_AGE,
    MIN_AGE,
    RACE_LABELS,
    ReferenceDatasetMetadata,
    ReferenceFilter,
)
from face_symmetry_lab.ui.i18n import t, translate_message
from face_symmetry_lab.ui.readings import PART_LABELS, part_label

_AGE_MODE_BAND: Final[str] = "年齢帯で選ぶ"
_AGE_MODE_RANGE: Final[str] = "1歳刻みの範囲で選ぶ"

#: Streamlit 既定の placeholder は英語（"Choose options"）である。日本語 UI の中に
#: そのまま残すと、3 つの絞り込みだけが別のアプリのように見える（§5.4）。
_MULTISELECT_PLACEHOLDER: Final[str] = "すべて（絞り込まない）"

#: AC-REFERENCE-01: 未選択の軸は絞り込みに使われず全件が対象になる。**その挙動を
#: 画面へ書く。** 書かないと、既定表示の件数が「全件」なのか「たまたま何かで
#: 絞られた数」なのかを読み取る手掛かりが画面上に無い。
#: §5.3.1: 常時表示は 1 行。絞り込みの規則（軸をまたぐと積、同じ軸は和）は
#: `help` へ移した——絞り込みを触っている人だけが必要とする情報である。
_FILTER_SCOPE_CAPTION: Final[str] = "未選択の軸は絞り込みに使いません（全件が対象になります）。"

_FILTER_SCOPE_HELP: Final[str] = (
    "軸をまたぐ条件は同時に満たすものだけが残り、"
    "同じ軸の複数選択はそのいずれかに一致するものが残ります。"
)

#: §8.6.7: 「点」は実測が集まる範囲へ合わせて引き伸ばした目盛りである。Analyze 側
#: （`ui.scores.SCORE_SCALE_CAPTION`）にだけ開示を置くと、同じ「点」を 2 つの画面で
#: 違う扱いにしてしまう。**§10.7 が禁じる語（順位・パーセンタイル・偏差値）を
#: 使わずに、母集団の中の位置を表さないことを述べる。**
_SCORE_SCALE_CAPTION: Final[str] = "点は引き伸ばした目盛りで、母集団の中の位置を表しません。"

_SCORE_SCALE_HELP: Final[str] = (
    "点は、実測が集まる範囲（ずれ 1.0〜5.0%）へ合わせて引き伸ばした目盛りで読んだ値です。"
    "引き伸ばした分だけばらつきも拡大します。"
)

#: §9.5 の 7 指標のうち、部位別・全体の平均対称度（%）として並べる 5 つ。
#: `domain.metrics.PART_NAMES` の順（輪郭・目・鼻・口）に全体を続ける——
#: `ui.readings` が使うラベルを再利用し、部位名の和訳を 2 か所に持たない。
_PERCENT_METRIC_LABELS: Final[tuple[tuple[str, str], ...]] = (
    *((f"symmetry_percent_{part}", PART_LABELS[part]) for part in PART_NAMES),
    ("symmetry_percent_overall", "全体"),
)

_FILTERS_HEADING: Final[str] = "絞り込み"
_RACE_LABEL: Final[str] = "人種"
_GENDER_LABEL: Final[str] = "性別"
_AGE_MODE_LABEL: Final[str] = "年齢の指定方法"
_AGE_BAND_LABEL: Final[str] = "年齢帯"
_AGE_RANGE_LABEL: Final[str] = "年齢の範囲（1歳刻み）"

_MATCHED_CELLS_EXPANDER_TEMPLATE: Final[str] = "一致したセル一覧（{count} 件）"
_NO_MATCHING_CELLS_CAPTION: Final[str] = "一致するセルがありません。"

_MATCHED_COUNT_LABEL: Final[str] = "一致した件数"
_NO_MATCH_NOTICE: Final[str] = (
    "条件に一致するデータがありません。人種・性別・年齢の条件を変更してください。"
)
_INSUFFICIENT_COUNT_TEMPLATE: Final[str] = (
    "一致した件数が {threshold} 件未満のため、平均・標準偏差は表示しません"
    "（件数が少ない集計は誤差が大きく、参考になりません）。"
)
_MEAN_SCORE_LABEL: Final[str] = "平均スコア（点）"
_SD_SCORE_LABEL: Final[str] = "標準偏差（点）"
_PART_BREAKDOWN_CAPTION: Final[str] = "部位別・全体の平均対称度（%）"
_OVERALL_LABEL: Final[str] = "全体"

_DISCLOSURES_HEADING: Final[str] = "この画面についての注意"
_DISCLOSURES_BODY: Final[str] = (
    "- この結果は探索値であり、"
    "正常値・目標値・個人評価・人物間比較には使用できません。\n"
    "- 群間の差は、撮影条件・注釈・構成比、"
    "ランドマーク検出の群ごとの挙動差を含みうるため、原因を断定できません。\n"
    "- 人種・性別・年齢は UTKFace の配布時の注釈であり、"
    "本アプリやモデルの推定ではありません。\n"
    "- この画面は平均と標準偏差だけを扱います。"
    "中央値と 95% 信頼区間はセル集計から復元できないため、v1 では表示しません。\n"
)
_DATASET_LICENSE_LINE_TEMPLATE: Final[str] = (
    "- {name} は非商用研究限定のデータセットです（{scope} · {url}）。"
)
_SOURCE_WARNING_CAPTION: Final[str] = "配布元の注意書き（原文）"

_VERSION_INFO_HEADING: Final[str] = "版とデータ出典"
_METRIC_VERSION_LINE_TEMPLATE: Final[str] = "- 指標の版: `{value}`"
_SCHEMA_VERSION_LINE_TEMPLATE: Final[str] = "- 参照データの schema: `{value}`"
_CELLS_SHA256_LINE_TEMPLATE: Final[str] = "- セル CSV の sha256: `{value}…`"
_LANDMARK_MAP_VERSION_LINE_TEMPLATE: Final[str] = "- ランドマーク対応表の版: `{value}`"

_PAGE_TITLE: Final[str] = "参照統計ブラウザ（UTKFace）"
_PAGE_CAPTION: Final[str] = (
    "探索的な記述統計です。正常値・目標値・個人評価・人物間比較には使えません。"
)
_PAGE_CAPTION_HELP: Final[str] = (
    "事前に集計した参照データセット（UTKFace、非商用研究限定）を表示します。"
    "この画面にはカメラ・画像の分析結果を表示しません。"
)
#: `resolved.unavailable_reason` は `ui.i18n.translate_message`（UI 境界専用）
#: で個別に訳す（`ReferenceDatasetError` が静的な文言のときだけ訳が効く）。
#: 外枠のこの template 自体は静的なので `t()` で訳す。
_UNAVAILABLE_WARNING_TEMPLATE: Final[str] = (
    "参照データを読み込めないため、この画面は利用できません（{reason}）。"
)
_RESULTS_HEADING: Final[str] = "結果"


def _age_mode_label(mode: str) -> str:
    """`st.radio` の `format_func`。**内部値（`age_mode` の比較に使う）は変えず**、
    表示だけを訳す（ADR-0010 (c)）。`t(mode)` のように lambda 引数を渡すと
    `tests/ui/test_i18n.py` が解決できないため、この module の module-level
    literal（`_AGE_MODE_BAND` / `_AGE_MODE_RANGE`）へ分岐してから `t()` へ渡す。
    """
    if mode == _AGE_MODE_BAND:
        return t(_AGE_MODE_BAND)
    return t(_AGE_MODE_RANGE)


@st.cache_resource
def _default_browser() -> ReferenceBrowser:
    """既定の参照データを 1 度だけ読み込み、以後の再実行では読み直さない。"""
    return ReferenceBrowser(CsvReferenceCellRepository())


def _render_filters() -> ReferenceFilter:
    """人種・性別・年齢の 3 軸。既定は未選択（= 絞り込まない、全件が対象）。"""
    st.subheader(t(_FILTERS_HEADING))
    st.caption(t(_FILTER_SCOPE_CAPTION), help=t(_FILTER_SCOPE_HELP))
    race_column, gender_column = st.columns(2)
    selected_races = race_column.multiselect(
        t(_RACE_LABEL),
        options=list(RACE_LABELS),
        placeholder=t(_MULTISELECT_PLACEHOLDER),
    )
    selected_genders = gender_column.multiselect(
        t(_GENDER_LABEL),
        options=list(GENDER_LABELS),
        placeholder=t(_MULTISELECT_PLACEHOLDER),
    )

    age_mode = st.radio(
        t(_AGE_MODE_LABEL),
        options=(_AGE_MODE_BAND, _AGE_MODE_RANGE),
        format_func=_age_mode_label,
        horizontal=True,
    )
    selected_age_bands: frozenset[str] = frozenset()
    age_range: tuple[int, int] | None = None
    if age_mode == _AGE_MODE_BAND:
        selected_age_bands = frozenset(
            st.multiselect(
                t(_AGE_BAND_LABEL),
                options=list(AGE_BANDS),
                placeholder=t(_MULTISELECT_PLACEHOLDER),
            )
        )
    else:
        age_range = st.slider(t(_AGE_RANGE_LABEL), MIN_AGE, MAX_AGE, (MIN_AGE, MAX_AGE))

    return ReferenceFilter(
        races=frozenset(selected_races),
        genders=frozenset(selected_genders),
        age_bands=selected_age_bands,
        age_range=age_range,
    )


def _render_matched_cells(result: ReferenceQueryResult) -> None:
    """一致したセル（race / gender / age / count）を一覧にする。

    **`st.dataframe`/pandas は使わない。** pandas 3.x の pyarrow-backed 文字列型
    への変換が、`AppTest` のスクリプト実行スレッドから呼ぶと segfault する環境が
    あるため、`ui` に前例のない依存を持ち込まず Markdown 表だけで表示する。
    """
    title = t(_MATCHED_CELLS_EXPANDER_TEMPLATE).format(count=len(result.matched_cells))
    with st.expander(title):
        if not result.matched_cells:
            st.caption(t(_NO_MATCHING_CELLS_CAPTION))
            return
        header = "| race | gender | age | count |\n|---|---|---|---|"
        rows = "\n".join(
            f"| {cell.race} | {cell.gender} | {cell.age} | {cell.count} |"
            for cell in result.matched_cells
        )
        st.markdown(f"{header}\n{rows}")


def _percent_metric_label(name: str) -> str:
    """`_PERCENT_METRIC_LABELS` の値を表示言語（FR-043）へ揃える。

    部位名は `ui.readings.part_label`（共有の訳付きアクセサ）へ委譲する。
    以前はここで `PART_LABELS[part]` の値を literal として複製していたが、
    `part_label` が同じ引き当てを既に提供しているため、複製をやめた。
    """
    if name == "symmetry_percent_overall":
        return t(_OVERALL_LABEL)
    part = name.removeprefix("symmetry_percent_")
    if part in PART_LABELS:
        return part_label(part)
    return dict(_PERCENT_METRIC_LABELS).get(name, name)


def _render_results(result: ReferenceQueryResult, *, min_group_for_ci: int) -> None:
    """§9.5: 件数を最初に見せ、しきい値未満では統計値を出さない（AC-REFERENCE-02）。"""
    aggregate = result.aggregate
    count = aggregate.count if aggregate is not None else 0
    st.metric(t(_MATCHED_COUNT_LABEL), f"{count:,}")

    if aggregate is None:
        st.info(t(_NO_MATCH_NOTICE))
    elif not aggregate.has_statistics:
        st.info(t(_INSUFFICIENT_COUNT_TEMPLATE).format(threshold=min_group_for_ci))
    else:
        score_summary = aggregate.metrics["score_points"]
        mean_column, sd_column = st.columns(2)
        mean_column.metric(t(_MEAN_SCORE_LABEL), f"{score_summary.mean:.2f}")
        sd_column.metric(
            t(_SD_SCORE_LABEL),
            "—" if score_summary.sd is None else f"{score_summary.sd:.2f}",
        )
        st.caption(t(_SCORE_SCALE_CAPTION), help=t(_SCORE_SCALE_HELP))

        st.caption(t(_PART_BREAKDOWN_CAPTION))
        # **5 つを `st.metric` へ分けない。** `st.metric` の値は 36px・
        # `white-space: nowrap`・`text-overflow: ellipsis` なので、5 列に割ると
        # 1 列 99px になり「95.05%」（110px）の末尾の `%` が省略記号へ置き換わる
        # （実測、画面幅に関わらず 5 つとも）。見出しの 3 つは 1〜2 列なので
        # `st.metric` のままでよい。表なら列幅に合わせて縮み、切れない。
        # `_render_matched_cells` と同じく pandas は使わない（同関数の docstring）。
        header = (
            "| "
            + " | ".join(_percent_metric_label(name) for name, _ in _PERCENT_METRIC_LABELS)
            + " |"
        )
        divider = "|" + "---|" * len(_PERCENT_METRIC_LABELS)
        values = (
            "| "
            + " | ".join(
                f"{aggregate.metrics[name].mean:.2f}%" for name, _ in _PERCENT_METRIC_LABELS
            )
            + " |"
        )
        st.markdown("\n".join((header, divider, values)))

    _render_matched_cells(result)


def _render_disclosures(metadata: ReferenceDatasetMetadata) -> None:
    """AC-REFERENCE-04: 探索的統計であること・注釈であって推定でないこと等。

    **文体はアプリの他の画面（ですます体）に揃える。** ここはアプリが利用者へ
    向けて書く文であり、下に続く `metadata.warning`（配布元の注意書き、原文の
    まま）とは出所が違う。同じ体裁で並べると、どちらがアプリの言葉でどちらが
    引用なのかが読み取れなくなる。
    """
    st.subheader(t(_DISCLOSURES_HEADING))
    st.markdown(
        t(_DISCLOSURES_BODY)
        + t(_DATASET_LICENSE_LINE_TEMPLATE).format(
            name=metadata.dataset_name,
            scope=metadata.license_scope,
            url=metadata.dataset_url,
        )
    )
    st.caption(t(_SOURCE_WARNING_CAPTION))
    st.text(metadata.warning)


def _render_version_info(metadata: ReferenceDatasetMetadata) -> None:
    """FR-020 と同じ作法で版と sha256 を出す。"""
    st.subheader(t(_VERSION_INFO_HEADING))
    st.markdown(
        "\n".join(
            (
                t(_METRIC_VERSION_LINE_TEMPLATE).format(value=metadata.metric_version),
                t(_SCHEMA_VERSION_LINE_TEMPLATE).format(value=metadata.schema_version),
                t(_CELLS_SHA256_LINE_TEMPLATE).format(value=metadata.cells_sha256[:16]),
                t(_LANDMARK_MAP_VERSION_LINE_TEMPLATE).format(value=metadata.landmark_map_version),
            )
        )
    )


def render_reference_page(browser: ReferenceBrowser | None = None) -> None:
    """参照統計ページ（FR-037）。`browser` は test 用の差し替え口。"""
    st.title(t(_PAGE_TITLE))
    # §5.3.1: 見出し直下は 1 行。3 文あった `st.info` を畳み、残りは `help` へ。
    st.caption(t(_PAGE_CAPTION), help=t(_PAGE_CAPTION_HELP))

    resolved = browser if browser is not None else _default_browser()

    if not resolved.is_available:
        # FR-043: 外枠の template は静的なので `t()` で訳す。埋め込む理由
        # （`ReferenceDatasetError` の文言）は `translate_message`（UI 境界専用）
        # で個別に訳を試みるが、値を埋め込んで組み立てられた文言は原文一致せず
        # 訳せないまま残る（既知の穴、ADR-0010 Consequences）。
        reason = resolved.unavailable_reason or ""
        st.warning(t(_UNAVAILABLE_WARNING_TEMPLATE).format(reason=translate_message(reason)))
        return

    metadata = resolved.metadata
    assert metadata is not None  # is_available が真なら必ず存在する

    query = _render_filters()
    result = resolved.query(query)
    if result is None:
        return

    st.subheader(t(_RESULTS_HEADING))
    _render_results(result, min_group_for_ci=metadata.min_group_for_ci)

    st.divider()
    _render_disclosures(metadata)
    _render_version_info(metadata)
