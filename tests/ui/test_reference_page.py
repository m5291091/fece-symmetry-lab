"""FR-037 / AC-REFERENCE-02〜04: 参照統計ページの描画。

このページの危険は数式ではなく**文脈**にある。同じ数字でも、注意書きが無ければ
「正常値」に見え、利用者自身の値と並べば「順位」に見える。したがって test の
主役は、数値の一致ではなく**何が画面に出て、何が出ていないか**である。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.reference_browser import ReferenceBrowser
from face_symmetry_lab.domain.reference_stats import (
    METRIC_NAMES,
    MetricMoments,
    ReferenceCell,
    ReferenceDataset,
    ReferenceDatasetMetadata,
)

_METADATA = ReferenceDatasetMetadata(
    schema_version="utkface-symmetry-3d-cells/1.0.0",
    metric_version="symmetry-3d/1.0.0",
    landmark_map_version="mediapipe-478-mirror/1.0.0",
    source_schema_version="utkface-symmetry-3d-labelled/3.0.0",
    source_generated_at="2026-08-14T03:38:57.977344+00:00",
    records_sha256="a" * 64,
    cells_sha256="b" * 64,
    total_count=40,
    min_group_for_ci=8,
    dataset_name="UTKFace",
    dataset_url="https://susanqq.github.io/UTKFace/",
    license_scope="non-commercial research only",
    selection={"age": "18-75", "max_abs_yaw_deg": "15.0"},
    warning=(
        "UTKFaceと固定した選別条件に限定した記述統計であり、探索値である。"
        "年齢・性別・人種ごとの平均は、対称度との相関・因果・優劣・人物の価値を示すものではない。"
        "群間の差はデータセットの撮影条件・注釈・構成比や、ランドマーク検出の群ごとの挙動差を含みうる。"
        "正常値・目標値・個人評価・人物間比較には使用できない。"
    ),
)


def _cell(race: str, gender: str, age: int, count: int, mean: float) -> ReferenceCell:
    return ReferenceCell(
        race=race,
        gender=gender,
        age=age,
        count=count,
        moments={name: MetricMoments(mean=mean, m2=float(count) * 4.0) for name in METRIC_NAMES},
    )


class _StubRepository:
    def __init__(self, dataset: ReferenceDataset | None) -> None:
        self._dataset = dataset

    def load(self) -> ReferenceDataset:
        if self._dataset is None:
            raise FileNotFoundError("configs/reference が無い")
        return self._dataset


def available_browser() -> ReferenceBrowser:
    """AppTest のスクリプトから読む helper（`tests.ui.conftest` と同じ作法）。"""
    return ReferenceBrowser(
        _StubRepository(
            ReferenceDataset(
                metadata=_METADATA,
                cells=(
                    _cell("Asian", "male", 30, 12, 40.0),
                    _cell("Asian", "female", 30, 10, 44.0),
                    _cell("White", "male", 45, 15, 48.0),
                    _cell("Others", "female", 60, 3, 52.0),
                ),
            )
        )
    )


def unavailable_browser() -> ReferenceBrowser:
    return ReferenceBrowser(_StubRepository(None))


def _run(builder: str = "available_browser") -> AppTest:
    script = (
        "from tests.ui.test_reference_page import " + builder + "\n"
        "from face_symmetry_lab.ui.reference_page import render_reference_page\n"
        "render_reference_page(" + builder + "())\n"
    )
    at = AppTest.from_string(script)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    """**アプリ自身が書いた文だけ**を集める。

    `st.text` は意図的に含めない。そこに出るのは配布元の注意書きの原文
    （`metadata.warning`）であり、「優劣…を示すものではない」のように否定文
    の中で価値判断語を含む。引用まで禁止語の走査へ入れると、原文を一字一句
    出すという AC-REFERENCE-04 の要求と衝突する。引用そのものは
    `test_reference_page_carries_the_source_warning_verbatim` が別に見る。
    """
    parts: list[str] = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.subheader, at.header, at.title):
        parts.extend(str(element.value) for element in block)
    parts.extend(str(element.label) for element in at.metric)
    parts.extend(str(element.value) for element in at.metric)
    return " ".join(parts)


def test_reference_page_renders_without_raising() -> None:
    """テスト 18: 描画の smoke。"""
    at = _run()

    assert not at.exception


def test_reference_page_offers_three_multiselect_axes() -> None:
    """FR-037: 人種・性別・年齢帯を複数選択でき、既定は未選択（= 全件）。"""
    at = _run()

    labels = " ".join(element.label for element in at.multiselect)

    assert "人種" in labels
    assert "性別" in labels
    assert "年齢" in labels
    assert all(element.value == [] for element in at.multiselect), "既定は未選択"


def test_reference_page_labels_the_filters_in_japanese() -> None:
    """日本語 UI の中に Streamlit 既定の英語 placeholder を残さない（§5.4）。

    罠: `placeholder=` を渡さない実装は、3 つの multiselect すべてに
    "Choose options" を出す。
    """
    at = _run()

    for element in at.multiselect:
        placeholder = element.proto.placeholder
        assert placeholder, f"{element.label} に placeholder が無い"
        assert not placeholder.isascii(), (
            f"{element.label} の placeholder が英語のまま: {placeholder}"
        )


def test_reference_page_says_that_an_unset_axis_matches_everything() -> None:
    """AC-REFERENCE-01: 未選択の軸は絞り込みに使われず全件が対象になる。

    その挙動が画面に書かれていないと、既定表示の件数が「全件」なのか
    「たまたま何かで絞られた数」なのか読み取れない。
    """
    at = _run()
    text = _all_text(at)

    assert "全件" in text
    assert "未選択" in text or "選択しない" in text


def test_the_part_breakdown_is_not_squeezed_into_five_metric_columns() -> None:
    """§5.4: 5 つ並べた `st.metric` は値が切れる。**実測の回帰。**

    `st.metric` の値は 36px・`white-space: nowrap`・`text-overflow: ellipsis`
    である。5 列に分けると 1 列 99px になり、「95.05%」（110px）が 11px はみ出して
    末尾の `%` が省略記号へ置き換わっていた——5 つとも、画面幅に関わらず。
    見出し（件数）は 1 列なので `st.metric` のままでよい。

    罠: 部位別を `st.metric` へ戻すと、この assertion が落ちる。
    """
    at = _run()

    labels = [element.label for element in at.metric]

    assert labels == ["一致した件数"]


def test_the_part_breakdown_still_shows_every_label_and_value() -> None:
    """並べ方を変えても、5 つの値は 1 つも減らさない。"""
    at = _run()
    text = _all_text(at)

    for label in ("輪郭", "目", "鼻", "口", "全体"):
        assert label in text
    # 4 セル（12+10+15 件のうち条件に一致する分）の平均が %表示で出ている。
    assert text.count("%") >= 5


def test_reference_page_shows_the_population_size_first() -> None:
    """FR-037: 件数を最初に見せる。全軸未選択なら全件が対象になる。"""
    at = _run()

    metric_labels = [element.label for element in at.metric]

    assert metric_labels, "件数を st.metric で出すこと"
    assert "件数" in metric_labels[0]
    assert at.metric[0].value.replace(",", "") == "40"


def test_reference_page_carries_the_source_warning_verbatim() -> None:
    """AC-REFERENCE-04: `summary.json` の warning をそのまま引く。

    **`st.text` の中身を直接見る。** 以前はこの test が `_all_text`（アプリ
    自身の文）だけを見ており、アプリ側の言い換えが原文と同じ語を含んでいた
    ために通っていた。引用が画面から丸ごと消えても気付けない状態だった。
    """
    at = _run()

    quoted = [str(element.value) for element in at.text]

    assert quoted == [_METADATA.warning], "配布元の注意書きを一字一句そのまま出すこと"


def test_reference_page_states_the_exploratory_limits_in_its_own_words() -> None:
    """AC-REFERENCE-04: 引用に頼らず、アプリ自身の言葉でも同じ制限を述べる。"""
    at = _run()
    text = _all_text(at)

    assert "探索値" in text
    assert "正常値・目標値・個人評価・人物間比較には使用できません" in text
    assert "群間の差" in text or "群ごとの挙動差" in text
    assert "非商用" in text or "non-commercial" in text
    assert "推定" in text and "注釈" in text, "注釈であって推定でないことを述べる"


def test_reference_page_shows_versions_and_hash() -> None:
    """AC-REFERENCE-04 / FR-020 の作法。"""
    at = _run()
    text = _all_text(at)

    assert "symmetry-3d/1.0.0" in text
    assert "utkface-symmetry-3d-cells/1.0.0" in text
    assert "b" * 16 in text, "セル CSV の sha256（先頭だけでも可）を出すこと"


def test_reference_page_states_it_shows_only_mean_and_sd() -> None:
    """§9.5: 中央値と CI は v1 で出さないことを明示する。"""
    at = _run()
    text = _all_text(at)

    assert "平均" in text and "標準偏差" in text
    assert "中央値" in text, "出さない理由を述べること"


def test_reference_page_never_relates_to_the_users_own_measurement() -> None:
    """§10.7 / AC-REFERENCE-04: 個人評価へ転用できる語を持たない。"""
    at = _run()
    text = _all_text(at)

    for forbidden in ("偏差値", "順位", "パーセンタイル", "平均より上", "平均より下", "あなたの"):
        assert forbidden not in text, f"個人評価に読める語が出ている: {forbidden}"


def test_reference_page_uses_no_value_judgement_words() -> None:
    """§10.7: 価値判断語を表示文言に使わない。"""
    at = _run()
    text = _all_text(at)

    for forbidden in ("美しい", "良い", "悪い", "優れ", "理想", "劣"):
        assert forbidden not in text, f"価値判断語が出ている: {forbidden}"


# --- §0 項目 34（撤回、2026-09-03）: 「全体のスコア」--------------------------


def test_the_reference_page_no_longer_shows_an_average_score() -> None:
    """§8.6.7 の表示スケールを撤回した。

    罠: 平均・標準偏差の `st.metric` ごと部位別・全体の % 表まで消す実装は、
    この test の後半（件数・部位別 4 つ・全体）で落ちる。
    """
    at = _run()
    text = _all_text(at)

    assert "平均スコア" not in text
    assert "標準偏差（点）" not in text

    metric_labels = [element.label for element in at.metric]
    assert "一致した件数" in metric_labels

    for label in ("輪郭", "目", "鼻", "口", "全体"):
        assert label in text


def test_reference_page_survives_missing_reference_data() -> None:
    """AC-REFERENCE-03: データが無くても例外にせず、理由を表示する。"""
    at = _run("unavailable_browser")

    assert not at.exception
    text = _all_text(at)
    assert "参照" in text
