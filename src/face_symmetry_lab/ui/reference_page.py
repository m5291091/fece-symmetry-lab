"""FR-037 / §9.5: 参照統計ブラウザ（人種・性別・年齢で絞り込む記述統計）。

`configs/reference/` に同梱した事前集計セルテーブルを、絞り込みと集計だけで
表示する。**この module は推論式・幾何計算・ファイル永続化を持たない。** 絞り込み
と集計は `application.reference_browser` / `domain.reference_stats` を呼ぶだけで
ある。

この画面が扱うのは UTKFace の配布時注釈（人種・性別・年齢）に対する記述統計で
あり、本アプリはこれらを推定しない（不変条件 7）。利用者自身の測定結果とは
一切結び付けない（§10.7）——偏差値・順位・パーセンタイル・「平均より上」「平均より下」を
作らず、両者を同じ画面へ並べない。

**`metadata.warning` は `st.text` で表示する。** `st.markdown`/`st.caption` は
Markdown として解釈するため、原文に記号が含まれると意図しない見た目になりうる。
`st.text` は書式解釈をしない、最も「そのまま」に近い widget である。
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
from face_symmetry_lab.ui.readings import PART_LABELS

_AGE_MODE_BAND: Final[str] = "年齢帯で選ぶ"
_AGE_MODE_RANGE: Final[str] = "1歳刻みの範囲で選ぶ"

#: Streamlit 既定の placeholder は英語（"Choose options"）である。日本語 UI の中に
#: そのまま残すと、3 つの絞り込みだけが別のアプリのように見える（§5.4）。
_MULTISELECT_PLACEHOLDER: Final[str] = "すべて（絞り込まない）"

#: AC-REFERENCE-01: 未選択の軸は絞り込みに使われず全件が対象になる。**その挙動を
#: 画面へ書く。** 書かないと、既定表示の件数が「全件」なのか「たまたま何かで
#: 絞られた数」なのかを読み取る手掛かりが画面上に無い。
_FILTER_SCOPE_CAPTION: Final[str] = (
    "未選択の軸は絞り込みに使いません（その軸は全件が対象になります）。"
    "軸をまたぐ条件は同時に満たすものだけが残り、同じ軸の複数選択はそのいずれかに一致するものが残ります。"
)

#: §8.6.7: 「点」は実測が集まる範囲へ合わせて引き伸ばした目盛りである。Analyze 側
#: （`ui.scores.SCORE_SCALE_CAPTION`）にだけ開示を置くと、同じ「点」を 2 つの画面で
#: 違う扱いにしてしまう。**§10.7 が禁じる語（順位・パーセンタイル・偏差値）を
#: 使わずに、母集団の中の位置を表さないことを述べる。**
_SCORE_SCALE_CAPTION: Final[str] = (
    "点は、実測が集まる範囲（ずれ 1.0〜5.0%）へ合わせて引き伸ばした目盛りで読んだ値です。"
    "引き伸ばした分だけばらつきも拡大します。母集団の中の位置を表す値ではありません。"
)

#: §9.5 の 7 指標のうち、部位別・全体の平均対称度（%）として並べる 5 つ。
#: `domain.metrics.PART_NAMES` の順（輪郭・目・鼻・口）に全体を続ける——
#: `ui.readings` が使うラベルを再利用し、部位名の和訳を 2 か所に持たない。
_PERCENT_METRIC_LABELS: Final[tuple[tuple[str, str], ...]] = (
    *((f"symmetry_percent_{part}", PART_LABELS[part]) for part in PART_NAMES),
    ("symmetry_percent_overall", "全体"),
)


@st.cache_resource
def _default_browser() -> ReferenceBrowser:
    """既定の参照データを 1 度だけ読み込み、以後の再実行では読み直さない。"""
    return ReferenceBrowser(CsvReferenceCellRepository())


def _render_filters() -> ReferenceFilter:
    """人種・性別・年齢の 3 軸。既定は未選択（= 絞り込まない、全件が対象）。"""
    st.subheader("絞り込み")
    st.caption(_FILTER_SCOPE_CAPTION)
    race_column, gender_column = st.columns(2)
    selected_races = race_column.multiselect(
        "人種",
        options=list(RACE_LABELS),
        placeholder=_MULTISELECT_PLACEHOLDER,
    )
    selected_genders = gender_column.multiselect(
        "性別",
        options=list(GENDER_LABELS),
        placeholder=_MULTISELECT_PLACEHOLDER,
    )

    age_mode = st.radio(
        "年齢の指定方法",
        options=(_AGE_MODE_BAND, _AGE_MODE_RANGE),
        horizontal=True,
    )
    selected_age_bands: frozenset[str] = frozenset()
    age_range: tuple[int, int] | None = None
    if age_mode == _AGE_MODE_BAND:
        selected_age_bands = frozenset(
            st.multiselect(
                "年齢帯",
                options=list(AGE_BANDS),
                placeholder=_MULTISELECT_PLACEHOLDER,
            )
        )
    else:
        age_range = st.slider("年齢の範囲（1歳刻み）", MIN_AGE, MAX_AGE, (MIN_AGE, MAX_AGE))

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
    with st.expander(f"一致したセル一覧（{len(result.matched_cells)} 件）"):
        if not result.matched_cells:
            st.caption("一致するセルがありません。")
            return
        header = "| race | gender | age | count |\n|---|---|---|---|"
        rows = "\n".join(
            f"| {cell.race} | {cell.gender} | {cell.age} | {cell.count} |"
            for cell in result.matched_cells
        )
        st.markdown(f"{header}\n{rows}")


def _render_results(result: ReferenceQueryResult, *, min_group_for_ci: int) -> None:
    """§9.5: 件数を最初に見せ、しきい値未満では統計値を出さない（AC-REFERENCE-02）。"""
    aggregate = result.aggregate
    count = aggregate.count if aggregate is not None else 0
    st.metric("一致した件数", f"{count:,}")

    if aggregate is None:
        st.info("条件に一致するデータがありません。人種・性別・年齢の条件を変更してください。")
    elif not aggregate.has_statistics:
        st.info(
            f"一致した件数が {min_group_for_ci} 件未満のため、平均・標準偏差は表示しません"
            "（件数が少ない集計は誤差が大きく、参考になりません）。"
        )
    else:
        score_summary = aggregate.metrics["score_points"]
        mean_column, sd_column = st.columns(2)
        mean_column.metric("平均スコア（点）", f"{score_summary.mean:.2f}")
        sd_column.metric(
            "標準偏差（点）",
            "—" if score_summary.sd is None else f"{score_summary.sd:.2f}",
        )
        st.caption(_SCORE_SCALE_CAPTION)

        st.caption("部位別・全体の平均対称度（%）")
        for column, (metric_name, label) in zip(
            st.columns(len(_PERCENT_METRIC_LABELS)),
            _PERCENT_METRIC_LABELS,
            strict=True,
        ):
            column.metric(label, f"{aggregate.metrics[metric_name].mean:.2f}%")

    _render_matched_cells(result)


def _render_disclosures(metadata: ReferenceDatasetMetadata) -> None:
    """AC-REFERENCE-04: 探索的統計であること・注釈であって推定でないこと等。

    **文体はアプリの他の画面（ですます体）に揃える。** ここはアプリが利用者へ
    向けて書く文であり、下に続く `metadata.warning`（配布元の注意書き、原文の
    まま）とは出所が違う。同じ体裁で並べると、どちらがアプリの言葉でどちらが
    引用なのかが読み取れなくなる。
    """
    st.subheader("この画面についての注意")
    st.markdown(
        "- この結果は探索値であり、"
        "正常値・目標値・個人評価・人物間比較には使用できません。\n"
        "- 群間の差は、撮影条件・注釈・構成比、"
        "ランドマーク検出の群ごとの挙動差を含みうるため、原因を断定できません。\n"
        "- 人種・性別・年齢は UTKFace の配布時の注釈であり、"
        "本アプリやモデルの推定ではありません。\n"
        "- この画面は平均と標準偏差だけを扱います。"
        "中央値と 95% 信頼区間はセル集計から復元できないため、v1 では表示しません。\n"
        f"- {metadata.dataset_name} は非商用研究限定のデータセットです"
        f"（{metadata.license_scope} · {metadata.dataset_url}）。"
    )
    st.caption("データセット配布元の注意書き（原文のまま）")
    st.text(metadata.warning)


def _render_version_info(metadata: ReferenceDatasetMetadata) -> None:
    """FR-020 と同じ作法で版と sha256 を出す。"""
    st.subheader("版とデータ出典")
    st.markdown(
        f"- 指標の版: `{metadata.metric_version}`\n"
        f"- 参照データの schema: `{metadata.schema_version}`\n"
        f"- セル CSV の sha256: `{metadata.cells_sha256[:16]}…`\n"
        f"- ランドマーク対応表の版: `{metadata.landmark_map_version}`"
    )


def render_reference_page(browser: ReferenceBrowser | None = None) -> None:
    """参照統計ページ（FR-037）。`browser` は test 用の差し替え口。"""
    st.title("参照統計ブラウザ（UTKFace）")
    st.caption("EXPLORATORY DESCRIPTIVE STATISTICS · NON-COMMERCIAL RESEARCH USE")
    st.info(
        "この画面は、事前に集計した参照データセットの探索的な記述統計を表示します。"
        "正常値・目標値・個人評価・人物間比較には使用できません。"
        "カメラ・画像の分析結果とは結び付けません。"
    )

    resolved = browser if browser is not None else _default_browser()

    if not resolved.is_available:
        st.warning(
            "参照データを読み込めないため、この画面は利用できません"
            f"（{resolved.unavailable_reason}）。"
            "カメラ・画像の左右差分析、参考年齢・参考性別、エクスポートには影響しません。"
        )
        return

    metadata = resolved.metadata
    assert metadata is not None  # is_available が真なら必ず存在する

    query = _render_filters()
    result = resolved.query(query)
    if result is None:
        return

    st.subheader("結果")
    _render_results(result, min_group_for_ci=metadata.min_group_for_ci)

    st.divider()
    _render_disclosures(metadata)
    _render_version_info(metadata)
