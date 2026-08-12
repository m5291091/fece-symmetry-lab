import re

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.domain.symmetry_3d import AxisSplit, Symmetry3DResult
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.scores import (
    DISTANCE_UNIT_CAPTION,
    GATE_COLORS,
    PART_EQUAL_WEIGHT_HELP,
    PART_EQUAL_WEIGHT_LABEL,
    PART_LEVEL_VARIATION_CAPTION,
    SCORE_HEADING,
    SYMMETRIZED_IMAGE_CAPTION,
    SYMMETRIZED_IMAGE_LIMITS,
    SYMMETRIZED_IMAGE_NOTICE,
    format_distance,
    format_symmetry,
    gate_color,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
    symmetry_percent,
)

#: §8.6.5: 部位ごとの `d_3d,p`/`s_3d` から `AxisSplit`/`Symmetry3DResult` を直接
#: 組み立てるテスト用 fixture。plane/depth の内訳はここでの表示テストには関係
#: しないため distance と同じ値を入れる。
_METRIC_VERSION_3D = "symmetry-3d/test"
_LANDMARK_MAP_VERSION = "map/test"


def _symmetry_3d_result(
    *,
    contour: float | None,
    eyes: float | None,
    nose: float | None,
    mouth: float | None,
    overall: float | None,
) -> Symmetry3DResult | None:
    """§8.6.5 の出力一式を、値だけから組み立てるテスト fixture。

    `overall=None` は「重ね合わせ自体が成立しない」（§8.5.1 の全欠測）を表す。
    """
    if overall is None:
        return None

    def _split(value: float | None) -> AxisSplit | None:
        return None if value is None else AxisSplit(distance=value, plane=value, depth=0.0)

    return Symmetry3DResult(
        splits={
            "contour": _split(contour),
            "eyes": _split(eyes),
            "nose": _split(nose),
            "mouth": _split(mouth),
        },
        overall=_split(overall),
        metric_version=_METRIC_VERSION_3D,
        landmark_map_version=_LANDMARK_MAP_VERSION,
    )


def _outcome_for_display(
    *,
    symmetry_3d: Symmetry3DResult | None,
    smooth_3d: float | None,
    symmetry_2d_overall: float | None = 0.5,
    part_smooth_3d: dict[str, float | None] | None = None,
) -> AnalysisOutcome:
    """§8.6.3 の表示テスト用 `AnalysisOutcome`。

    2D の値（既定 0.5）は 3D の値（テストでは概ね 0.0〜0.1）から大きく離す。
    表示ロジックが誤って 2D を読んでいれば、この差で assertion が落ちる
    （fixture 設計上の罠、いちばん重要な部分）。``part_smooth_3d`` は §8.7 改訂
    （H1）の部位別平滑値で、既定 `None`（呼び出し側の多くは生値へのフォール
    バックだけを確かめるため、指定しなくても動く）。
    """
    symmetry_2d = None
    if symmetry_2d_overall is not None:
        symmetry_2d = SymmetryResult(
            distances={
                "contour": symmetry_2d_overall,
                "eyes": symmetry_2d_overall,
                "nose": symmetry_2d_overall,
                "mouth": symmetry_2d_overall,
            },
            overall=symmetry_2d_overall,
            metric_version="symmetry-2d/test",
            landmark_map_version=_LANDMARK_MAP_VERSION,
        )
    return AnalysisOutcome(
        frame_id=1,
        state=AnalysisState(status=Status.OK),
        symmetry=symmetry_2d,
        overall_distance_smooth=symmetry_2d_overall,
        pose=None,
        face_area_ratio=None,
        metric_version="symmetry-2d/test",
        landmark_map_version=_LANDMARK_MAP_VERSION,
        symmetry_3d=symmetry_3d,
        overall_distance_3d_smooth=smooth_3d,
        part_distances_3d_smooth=part_smooth_3d,
    )


def _run(body: str) -> AppTest:
    at = AppTest.from_string(
        "from face_symmetry_lab.domain.landmarks import TrackingStatus\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    render_tracking_status,\n"
        ")\n"
        "MESSAGES = {status: status.name.lower().replace('_', ' ') for status in TrackingStatus}\n"
        f"{body}\n"
    )
    at.run()
    return at


def _headline_metric_rows(at: AppTest) -> list[list[str]]:
    """`render_headline_metrics` が作る `st.columns` のブロックごとに、含まれる
    metric ラベルを描画順に取り出す。

    §5.1・不変条件 5（§1.3.1）が求める「左右差の行」と「参考値の行」が実際に
    分かれていることを、Streamlit の要素木から直接確認するためのヘルパー。
    1 列に並ぶ順序だけでは行が分かれているかを示せないため、`st.columns` の
    呼び出しごとに独立した `flex_container` ブロックが生まれることを利用する。
    """
    rows: list[list[str]] = []
    for child in at.main.children.values():
        if getattr(child, "type", None) != "flex_container":
            continue
        labels = [
            metric.label
            for column in child.children.values()
            for metric in column.children.values()
            if getattr(metric, "type", None) == "metric"
        ]
        if labels:
            rows.append(labels)
    return rows


@pytest.mark.parametrize(
    ("status", "severity"),
    [
        (TrackingStatus.ERROR, "error"),
        (TrackingStatus.MULTI_FACE, "warning"),
        (TrackingStatus.SINGLE_FACE, "success"),
        (TrackingStatus.NO_FACE, "info"),
    ],
)
def test_each_tracking_state_renders_at_its_assigned_severity(
    status: TrackingStatus,
    severity: str,
) -> None:
    """§5.2 が状態ごとの色を固定するので、Camera と Image が食い違うことはない。"""
    at = _run(f"render_tracking_status(TrackingStatus.{status.name}, MESSAGES)")

    expected = status.name.lower().replace("_", " ")
    assert not at.exception
    assert [element.value for element in getattr(at, severity)] == [expected]


def test_missing_headline_values_render_as_a_dash_rather_than_a_number() -> None:
    """NFR-013: 欠測は欠測として示し、古い値や 0 として見せない。"""
    at = _run("render_headline_metrics(overall=None, equal_weight=None, estimate=None)")

    assert not at.exception
    assert [element.value for element in at.metric] == ["—", "—", "—", "—"]


def test_the_detail_panel_words_reasons_from_the_shared_guidance_table() -> None:
    """§5.3/§6.3: パネルとライブのバナーは同じ表を読むので、理由コードの文言は
    1 度だけ書かれる。さらに `test_guidance.py` の全網羅チェックにより、新しく
    追加したコードに対してこのパネルが黙って何も描かない事態を防げる。"""
    at = AppTest.from_string(
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status\n"
        "from face_symmetry_lab.ui.scores import render_guidance\n"
        "render_guidance(\n"
        "    AnalysisOutcome(\n"
        "        frame_id=1,\n"
        "        state=AnalysisState(\n"
        "            status=Status.HOLD,\n"
        "            reason=ReasonCode.TOO_FAR,\n"
        "            reason_codes=(ReasonCode.TOO_FAR, ReasonCode.OFF_AXIS_YAW),\n"
        "        ),\n"
        "        symmetry=None,\n"
        "        overall_distance_smooth=None,\n"
        "        pose=None,\n"
        "        face_area_ratio=None,\n"
        "        metric_version='metric/test',\n"
        "        landmark_map_version='map/test',\n"
        "    )\n"
        ")\n"
    )
    at.run()

    rendered = " ".join(element.value for element in at.markdown)
    assert not at.exception
    assert REASON_GUIDANCE[ReasonCode.TOO_FAR] in rendered
    assert REASON_GUIDANCE[ReasonCode.OFF_AXIS_YAW] in rendered


# --- §8.6.3 対称度パーセントの表示 ------------------------------------------


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, "100.00%"),
        (0.0215, "97.85%"),
        (0.5, "50.00%"),
        (1.0, "0.00%"),
    ],
)
def test_a_distance_is_shown_as_its_complement(distance: float, expected: str) -> None:
    """§8.6.3: `100% - ずれ%`, with no coefficient anywhere in the transform."""
    assert format_symmetry(distance) == expected


def test_the_symmetry_and_deviation_readings_always_sum_to_one_hundred() -> None:
    """AC-DISPLAY-02: 2 つの向きは同じ測定値である。"""
    for distance in (0.0, 0.004, 0.0215, 0.31, 0.999):
        symmetry = float(format_symmetry(distance).rstrip("%"))
        deviation = float(format_distance(distance).rstrip("%"))
        assert symmetry + deviation == pytest.approx(100.0, abs=0.02)


def test_a_deviation_beyond_the_interocular_distance_floors_at_zero() -> None:
    """§8.6.3 がクランプするのは *表示* だけで、`d` 自体は上限を持たない。"""
    assert format_symmetry(4.0) == "0.00%"
    assert format_distance(4.0) == "400.00%"


def test_a_missing_distance_has_no_symmetry_reading() -> None:
    """NFR-013: 欠測が「完全に対称な 100%」と読めてはならない。"""
    assert format_symmetry(None) == "—"


def test_the_headline_leads_with_symmetry_and_still_states_the_deviation() -> None:
    """§5.3: 両方の向きを画面に出すので、100% が目標として通ることはない。"""
    at = AppTest.from_string(
        "from face_symmetry_lab.ui.scores import render_headline_metrics\n"
        "render_headline_metrics(overall=0.0215, equal_weight=None, estimate=None)\n"
    )
    at.run()

    assert not at.exception
    assert at.metric[0].label == "全体の対称度"
    assert at.metric[0].value == "97.85%"
    captions = " ".join(element.value for element in at.caption)
    assert "2.15%" in captions


def test_the_part_equal_weight_headline_label_uses_the_user_specified_wording() -> None:
    """ユーザー指定により「部位平均の対称度」ではなく「全体のスコア」を見出しの
    ラベルにする（2026-08-11、W5-fix）。"""
    assert PART_EQUAL_WEIGHT_LABEL == "全体のスコア"


def test_symmetry_and_reference_metrics_render_on_separate_headline_rows() -> None:
    """不変条件 5（§1.3.1）: 参考性別・参考年齢は、左右差（全体の対称度・全体の
    スコア）と混ざらない別の行に置く（§5.1 の 2 行構成）。1 列に並ぶ順序だけでは
    確認できないため、`st.columns` のブロックが実際に分かれていることを要素木
    から直接確認する。"""
    at = _run("render_headline_metrics(overall=0.02, equal_weight=0.05, estimate=None)")

    assert not at.exception
    rows = _headline_metric_rows(at)
    assert rows == [["全体の対称度", PART_EQUAL_WEIGHT_LABEL], ["参考性別", "参考年齢"]]


def test_the_symmetry_heading_and_caption_avoid_a_value_judgement() -> None:
    """§5.2/§5.3: reorienting the number must not turn it into a grade.

    Checked against the strings the user actually reads as a verdict — the
    heading and the unit caption — rather than the whole module, so an
    explicit disclaimer ("良し悪しを表しません") cannot fail this.
    """
    for forbidden in ("スコア", "得点", "評価", "優秀", "理想", "完璧"):
        assert forbidden not in SCORE_HEADING
        assert forbidden not in DISTANCE_UNIT_CAPTION

    # 注記は 100% が何であるかを述べ続けなければならない。さもないと、向きを
    # 変えただけのものが、利用者の目指す目標へ黙って変わってしまう。
    assert "上限" in DISTANCE_UNIT_CAPTION
    assert "目標値ではありません" in DISTANCE_UNIT_CAPTION

    for forbidden in ("スコア", "得点", "評価", "優秀", "理想", "完璧"):
        assert forbidden not in PART_LEVEL_VARIATION_CAPTION
        assert forbidden not in PART_EQUAL_WEIGHT_HELP


def test_the_part_equal_weight_help_explains_the_term_is_not_a_calibrated_score() -> None:
    """§8.6.3 の『表示語としてのスコア』: ラベルは「全体のスコア」だが、help は
    値の実体（部位を 1 票ずつ数えた対称度）と、採点ではないことの両方を明示する。
    ラベル自体は「スコア」を含むが、help はその語を使わずに採点ではないと言える
    ことを確認する（既存の禁止語チェックと矛盾しない形にするため）。"""
    assert "部位を 1 票ずつ" in PART_EQUAL_WEIGHT_HELP
    assert "採点" in PART_EQUAL_WEIGHT_HELP
    for forbidden in ("優秀", "理想", "完璧", "得点", "評価", "スコア"):
        assert forbidden not in PART_EQUAL_WEIGHT_HELP


# --- §8.6.3/§8.6.5: 表示・エクスポートを 3D 指標へ切り替える -----------------


def test_the_resolved_overall_distance_prefers_the_three_dimensional_smoothed_value() -> None:
    """§8.6.3/§8.7: 主表示は 3D の平滑値であり、2D の値ではない（罠 (a)）。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
        symmetry_2d_overall=0.5,
    )

    resolved = resolve_overall_distance(outcome)

    assert resolved == pytest.approx(0.021)
    assert resolved != pytest.approx(0.5)


def test_the_resolved_overall_distance_falls_back_to_the_raw_three_dimensional_value() -> None:
    """EMA が未確立（生値のみ）でも欠測にせず、3D の生値を使う。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=None,
    )

    assert resolve_overall_distance(outcome) == pytest.approx(0.018)


def test_the_resolved_overall_distance_is_missing_without_a_three_dimensional_result() -> None:
    """NFR-013: 3D 一式が欠測なら `—`（`None`）であり、0 や 2D の値で埋めない。"""
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None, symmetry_2d_overall=0.5)

    assert resolve_overall_distance(outcome) is None


def test_the_resolved_equal_weight_distance_always_reads_the_raw_value() -> None:
    """§8.6.5/§8.7: `s_3d` は EMA の対象に含めないため、`resolve_equal_weight_distance`
    は常に生値を読む（`resolve_overall_distance` と異なり、平滑値へは決して
    フォールバックしない——そもそも平滑値を持たない）。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )

    resolved = resolve_equal_weight_distance(outcome)

    assert resolved is not None
    assert resolved != pytest.approx(resolve_overall_distance(outcome))


def test_the_resolved_equal_weight_distance_is_missing_without_a_three_dimensional_result() -> None:
    """NFR-013: 3D 一式が欠測なら `—`（`None`）であり、0 で埋めない。"""
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)

    assert resolve_equal_weight_distance(outcome) is None


# --- §8.7 改訂（H1）: 部位別 (`d_3d,p`) の平滑値表示 -------------------------


def test_symmetry_percent_agrees_with_the_formatted_string() -> None:
    """`symmetry_percent` は `format_symmetry` と同じ変換を数値で返す
    （HUD の折れ線グラフがテキストと別の表示規則へ分岐しないための共有経路）。"""
    for distance in (0.0, 0.0215, 0.5, 1.0, 4.0):
        percent = symmetry_percent(distance)
        assert percent is not None
        assert f"{percent:.2f}%" == format_symmetry(distance)


def test_symmetry_percent_of_a_missing_value_is_missing() -> None:
    assert symmetry_percent(None) is None


def test_resolve_part_distance_prefers_the_smoothed_value_over_the_raw_one() -> None:
    """§8.7 改訂: 部位別も全体値と同じく「平滑値を優先し、無ければ生値」の
    解決を通す。「常に生値を読む実装」ではこの assertion が落ちる（罠）。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.05, eyes=0.05, nose=0.05, mouth=0.05, overall=0.05
        ),
        smooth_3d=0.05,
        part_smooth_3d={"contour": 0.02, "eyes": 0.02, "nose": 0.02, "mouth": 0.02},
    )

    resolved = resolve_part_distance_3d(outcome, "contour")

    assert resolved == pytest.approx(0.02)
    assert resolved != pytest.approx(0.05)


def test_resolve_part_distance_falls_back_to_raw_when_the_smoothed_container_is_missing() -> None:
    """静止画（§8.7 改訂、`part_distances_3d_smooth is None`）は生値へ落ちる。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=None,
        part_smooth_3d=None,
    )

    assert resolve_part_distance_3d(outcome, "nose") == pytest.approx(0.03)


def test_resolve_part_distance_falls_back_per_part_when_only_that_key_is_missing() -> None:
    """部位が個別に欠測(不変条件 11)しても、他の部位は平滑値を読み続ける。"""
    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.018,
        part_smooth_3d={"contour": None, "eyes": 0.09, "nose": 0.09, "mouth": 0.09},
    )

    assert resolve_part_distance_3d(outcome, "contour") == pytest.approx(0.01)  # 生値へ回帰
    assert resolve_part_distance_3d(outcome, "eyes") == pytest.approx(0.09)  # 平滑値のまま


def test_resolve_part_distance_is_missing_without_a_three_dimensional_result() -> None:
    """NFR-013: 3D 一式が欠測なら `—`（`None`）であり、0 で埋めない。"""
    outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)

    assert resolve_part_distance_3d(outcome, "mouth") is None


def test_the_part_breakdown_reads_from_the_three_dimensional_result() -> None:
    """§8.6.3: 部位別 4 部位は `d_3d,p` から作られる（2D の値ではない）。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        "    symmetry_2d_overall=0.5,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["輪郭"] == format_symmetry(0.01)
    assert values["目"] == format_symmetry(0.02)
    assert values["鼻"] == format_symmetry(0.03)
    assert values["口"] == format_symmetry(0.015)
    for part_value in (values["輪郭"], values["目"], values["鼻"], values["口"]):
        assert part_value != format_symmetry(0.5)


def test_the_part_breakdown_shows_the_smoothed_value_as_primary_and_notes_the_raw_one() -> None:
    """§8.7 改訂（H1）: 部位別の平滑値を持つときは、詳細タブも全体値と同じ
    「平滑値を主表示し生値を併記する」構成にする（表示の一貫性、片方が生値・
    片方が平滑値という 2 つの表示規則を作らない）。生値をそのまま出す実装では
    ``values["輪郭"]`` の assertion が落ちる（罠）。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        "    part_smooth_3d={\n"
        "        'contour': 0.004, 'eyes': 0.006, 'nose': 0.009, 'mouth': 0.005,\n"
        "    },\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=True)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    # 主表示は平滑値であり、生値(0.01 等)そのものではない。
    assert values["輪郭"] == format_symmetry(0.004)
    assert values["目"] == format_symmetry(0.006)
    assert values["鼻"] == format_symmetry(0.009)
    assert values["口"] == format_symmetry(0.005)
    # 生値は併記される(全体値の「平滑化前の生値」と同じ構成)。
    captions = " ".join(element.value for element in at.caption)
    for raw in (format_symmetry(0.01), format_symmetry(0.02), format_symmetry(0.03)):
        assert raw in captions


def test_the_headline_part_equal_weight_score_is_a_distinct_reading_from_the_overall() -> None:
    """§8.6.5: 見出しの「全体のスコア」（`s_3d`）は「全体の対称度」（`d_3d`）と
    重みだけが違う別の総合量で、一致しない。ここへ `d_3d` を誤って渡すと
    （`equal_weight=resolve_overall_distance(outcome)`）この assertion が
    落ちることを確認済み。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    resolve_equal_weight_distance,\n"
        "    resolve_overall_distance,\n"
        ")\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.005, eyes=0.005, nose=0.06, mouth=0.005, overall=0.02\n"
        "    ),\n"
        "    smooth_3d=0.02,\n"
        ")\n"
        "render_headline_metrics(\n"
        "    overall=resolve_overall_distance(outcome),\n"
        "    equal_weight=resolve_equal_weight_distance(outcome),\n"
        "    estimate=None,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["全体の対称度"] == format_symmetry(0.02)
    assert values[PART_EQUAL_WEIGHT_LABEL] != values["全体の対称度"]


def test_the_headline_score_is_missing_when_a_single_part_is_missing_but_overall_remains() -> None:
    """§8.6.5: 部位が 1 つ欠測しても、見出しの「全体のスコア」だけが `—` になり、
    「全体の対称度」は出続ける（AC-SCORE-10 と同じ非対称な欠測条件）。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    resolve_equal_weight_distance,\n"
        "    resolve_overall_distance,\n"
        ")\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=None, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_headline_metrics(\n"
        "    overall=resolve_overall_distance(outcome),\n"
        "    equal_weight=resolve_equal_weight_distance(outcome),\n"
        "    estimate=None,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["全体の対称度"] == format_symmetry(0.018)
    assert values[PART_EQUAL_WEIGHT_LABEL] == "—"


def test_a_single_missing_part_only_shows_that_parts_detail_reading_as_a_dash() -> None:
    """§8.6.5: 部位が 1 つ欠測しても、詳細タブの他の部位は出続ける
    （AC-SCORE-10 と同じ非対称な欠測条件）。「全体のスコア」の欠測は見出し側の
    テストで確認する——詳細タブにはもう描画しないため、ここには現れない。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=None, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["鼻"] == "—"
    assert values["輪郭"] == format_symmetry(0.01)
    assert values["目"] == format_symmetry(0.02)
    assert values["口"] == format_symmetry(0.015)
    assert PART_EQUAL_WEIGHT_LABEL not in values


def test_the_headline_shows_both_dashes_when_the_three_dimensional_result_is_missing() -> None:
    """不変条件 11: §8.5.1 の重ね合わせが成立しないとき、見出しの「全体の対称度」
    「全体のスコア」はどちらも 0 や前値ではなく `—`。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    resolve_equal_weight_distance,\n"
        "    resolve_overall_distance,\n"
        ")\n"
        "outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)\n"
        "render_headline_metrics(\n"
        "    overall=resolve_overall_distance(outcome),\n"
        "    equal_weight=resolve_equal_weight_distance(outcome),\n"
        "    estimate=None,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["全体の対称度"] == "—"
    assert values[PART_EQUAL_WEIGHT_LABEL] == "—"


def test_a_missing_three_dimensional_result_shows_every_part_reading_as_a_dash() -> None:
    """不変条件 11: §8.5.1 の重ね合わせが成立しないとき、詳細タブの部位別は
    すべて `—`（0 や前値ではない）。見出しの全体の対称度・全体のスコアは
    別テストで確認する。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    for label in ("輪郭", "目", "鼻", "口"):
        assert values[label] == "—"
    assert PART_EQUAL_WEIGHT_LABEL not in values


def test_the_detail_panel_no_longer_duplicates_the_headline_score() -> None:
    """全体のスコアは見出しへ表示するため、詳細タブでは重複して描画しない。
    部位別 4 つと `PART_LEVEL_VARIATION_CAPTION`（揺れ幅の注記）は詳細タブに
    残る。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    labels = [metric.label for metric in at.metric]
    assert PART_EQUAL_WEIGHT_LABEL not in labels
    for expected in ("輪郭", "目", "鼻", "口"):
        assert expected in labels
    captions = " ".join(element.value for element in at.caption)
    assert PART_LEVEL_VARIATION_CAPTION in captions


def test_the_part_level_variation_note_applies_equally_to_all_four_parts() -> None:
    """§5.3: 撮影のたびに動く幅より小さい変化は読み取れないという注記を、特定の
    部位（鼻など）だけを名指しして述べない。"""
    for singled_out in ("鼻だけ", "鼻のみ", "鼻は使えません", "鼻は信頼できません"):
        assert singled_out not in PART_LEVEL_VARIATION_CAPTION
    assert "撮影" in PART_LEVEL_VARIATION_CAPTION


# --- 項目26（H3）: Camera の詳細タブから重複数値を除く（show_parts） --------


def test_show_parts_false_hides_the_part_metrics_but_keeps_the_required_notices() -> None:
    """項目26（H3）: Camera は HUD が既に部位別 4 つを表示するため描かないが、
    §5.3 が義務付ける揺れ幅の注記と H2 の計算過程は詳細タブが唯一の受け皿なので
    描き続ける。罠: 「部位別セクションを丸ごと削る」実装だと、注記と計算過程も
    一緒に消えてこの assertion が落ちる。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False, show_parts=False)\n"
    )
    at.run()

    assert not at.exception
    labels = [metric.label for metric in at.metric]
    for hidden in ("輪郭", "目", "鼻", "口"):
        assert hidden not in labels

    headings = [element.value for element in at.subheader]
    assert SCORE_HEADING in headings
    infos = " ".join(element.value for element in at.info)
    assert "試験中の指標" in infos

    captions = " ".join(element.value for element in at.caption)
    assert PART_LEVEL_VARIATION_CAPTION in captions

    walkthrough = " ".join(element.value for element in at.code)
    for label in ("輪郭", "目", "鼻", "口"):
        assert label in walkthrough


def test_show_parts_false_still_shows_pose_and_metric_version_captions() -> None:
    """HUD が持たない値（姿勢・指標の版）は `show_parts=False` でも描き続ける。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False, show_parts=False)\n"
    )
    at.run()

    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    # `_outcome_for_display` は pose=None を渡すため、render_pose の None 分岐の
    # 文言が出る。値の有無に関わらず render_pose 自体が呼ばれたことの確認になる。
    assert "姿勢: 取得できません" in captions
    assert f"metric {_METRIC_VERSION_3D}" in captions


def test_render_score_details_show_parts_defaults_to_true() -> None:
    """Image はこの既定値のまま呼ぶ。既定を `False` に変えると Image 側からも
    部位別が消えるため、パラメータの既定値そのものを固定する。"""
    import inspect

    from face_symmetry_lab.ui.scores import render_score_details

    signature = inspect.signature(render_score_details)

    assert signature.parameters["show_parts"].default is True


# --- §8.6.3(H2): 「全体のスコア」の計算過程の開示 ----------------------------


def test_the_calculation_walkthrough_shows_the_four_parts_and_the_final_result() -> None:
    """H2: 計算過程に、部位別 4 つの値・RMS の途中経過・最終値が出る。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    walkthrough = " ".join(element.value for element in at.code)
    for label in ("輪郭", "目", "鼻", "口"):
        assert label in walkthrough
    for deviation in ("2.11", "1.44", "0.62", "1.83"):
        assert deviation in walkthrough
    assert "1.60" in walkthrough  # sqrt(mean(2.11²+1.44²+0.62²+1.83²)/4)
    assert "98.40" in walkthrough  # 100 - 1.60


def test_the_calculation_walkthrough_matches_the_part_level_metrics_above_it() -> None:
    """H2: 計算過程の部位別の値は、上の部位別メトリクス（主表示、H1 の平滑値）と
    文字列として一致する。罠 (b): 計算過程だけ生値（1.00/2.00/3.00/1.50）を使うと
    ここが落ちる。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        "    part_smooth_3d={\n"
        "        'contour': 0.004, 'eyes': 0.006, 'nose': 0.009, 'mouth': 0.005,\n"
        "    },\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=True)\n"
    )
    at.run()

    assert not at.exception
    metric_values = {metric.label: metric.value for metric in at.metric}
    walkthrough = " ".join(element.value for element in at.code)
    for label, smoothed in (("輪郭", 0.004), ("目", 0.006), ("鼻", 0.009), ("口", 0.005)):
        assert metric_values[label] == format_symmetry(smoothed)
        assert format_distance(smoothed).rstrip("%") in walkthrough
    for raw_deviation in ("1.00", "2.00", "3.00", "1.50"):
        assert raw_deviation not in walkthrough


def test_the_calculation_walkthrough_matches_resolve_part_distance_when_hidden() -> None:
    """AC-DISPLAY-04: Camera（`show_parts=False`）では部位別メトリクスが描かれ
    ないため、上の表示とは比較できない。代わりに、計算過程が使う値が
    `resolve_part_distance_3d`（HUD が `format_symmetry` を通す前と同じ値）から
    直接一致することを確認する。罠: 計算過程だけ別の値（例えば生値）を使う実装
    はこの assertion が落ちる。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        "    part_smooth_3d={\n"
        "        'contour': 0.004, 'eyes': 0.006, 'nose': 0.009, 'mouth': 0.005,\n"
        "    },\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False, show_parts=False)\n"
    )
    at.run()

    assert not at.exception
    assert not at.metric  # HUD が既に部位別を表示しているため、ここには無い

    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.018,
        part_smooth_3d={"contour": 0.004, "eyes": 0.006, "nose": 0.009, "mouth": 0.005},
    )
    walkthrough = " ".join(element.value for element in at.code)
    for name in ("contour", "eyes", "nose", "mouth"):
        value = resolve_part_distance_3d(outcome, name)
        assert format_distance(value).rstrip("%") in walkthrough


def test_the_calculation_walkthrough_shows_no_numbers_when_a_part_is_missing() -> None:
    """H2/不変条件 11: `s_3d` が部位欠測で欠測のとき、計算過程は数値を代入しない
    （罠: 欠測を 0 へ fallback すると `100.00` という偽の満点が出てしまう）。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=None, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.018,\n"
        ")\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    walkthrough = " ".join(element.value for element in at.code)
    assert "100.00" not in walkthrough
    assert "1.00" not in walkthrough  # 輪郭のずれを代入した行そのものが無い
    notices = " ".join(element.value for element in list(at.info) + list(at.caption))
    assert "当てはめられません" in notices


def test_the_calculation_walkthrough_wording_avoids_value_judgements_and_thresholds() -> None:
    """§5.3/§10.7/不変条件 8: 計算過程の説明文にも他の文言と同じ禁止語が適用され、
    加えて母集団の分布を持たないため基準値も書かない。"""
    from face_symmetry_lab.ui.scores import (
        SCORE_CALCULATION_DEVIATION_INTRO,
        SCORE_CALCULATION_HIGH_VALUE_NOTE,
        SCORE_CALCULATION_MEASURED_RANGE_CAPTION,
        SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION,
    )

    combined = (
        SCORE_CALCULATION_DEVIATION_INTRO
        + SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION
        + SCORE_CALCULATION_HIGH_VALUE_NOTE
        + SCORE_CALCULATION_MEASURED_RANGE_CAPTION
    )
    for forbidden in (
        "良い",
        "悪い",
        "美しい",
        "優秀",
        "優れ",
        "完璧",
        "高得点",
        "低得点",
        "満点",
        "合格",
    ):
        assert forbidden not in combined
    for threshold_phrase in ("以上なら対称", "%以上で対称", "以上あれば対称"):
        assert threshold_phrase not in combined


def test_the_calculation_walkthrough_notes_the_single_subject_caveat() -> None:
    """OD-09: 個人差は未検証であるため「被写体 1 名」の但し書きを必ず書く。"""
    from face_symmetry_lab.ui.scores import SCORE_CALCULATION_MEASURED_RANGE_CAPTION

    assert "被写体 1 名" in SCORE_CALCULATION_MEASURED_RANGE_CAPTION


# --- FR-035 対称化画像の枠付け ----------------------------------------------


def _warp_view(*, warped, reason=None):
    """描画側が読む項目だけを持つ `SymmetrizationView` のスタブ。"""
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class _View:
        warped: Any
        warp_unavailable_reason: Any

    return _View(warped=warped, warp_unavailable_reason=reason)


def test_the_warped_image_is_not_rendered_unless_it_was_requested() -> None:
    """§8.6.4 は任意指定である。要求されていない変形については何も述べない。"""
    at = AppTest.from_string(
        "import numpy as np, streamlit as st\n"
        "from face_symmetry_lab.ui.scores import render_symmetrized_image\n"
        "from tests.ui.test_scores import _warp_view\n"
        "render_symmetrized_image(_warp_view(warped=None), requested=False)\n"
    )
    at.run()

    assert not at.exception
    assert not at.info
    assert not at.image


def test_a_refused_warp_explains_itself_instead_of_rendering_nothing() -> None:
    """AC-SYMM-02: 理由を画面に出したうえで fail-soft する。"""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.scores import render_symmetrized_image\n"
        "from tests.ui.test_scores import _warp_view\n"
        "render_symmetrized_image(\n"
        "    _warp_view(warped=None, reason='変形量が大きすぎるため画像を生成しません。'),\n"
        "    requested=True,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert any("変形量" in element.value for element in at.info)


def test_the_warped_image_is_labelled_as_a_manipulated_image() -> None:
    """§5.3/§10.7: 対称化した写真が、無加工の写真として通ってはならない。"""
    at = AppTest.from_string(
        "import numpy as np\n"
        "from face_symmetry_lab.ui.scores import render_symmetrized_image\n"
        "from tests.ui.test_scores import _warp_view\n"
        "render_symmetrized_image(\n"
        "    _warp_view(warped=np.zeros((8, 8, 3), dtype=np.uint8)),\n"
        "    requested=True,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.image
    rendered = " ".join(element.value for element in at.caption)
    assert "加工画像" in rendered
    assert "実際の顔写真ではなく" in rendered


def test_the_warped_image_wording_avoids_a_value_judgement() -> None:
    """§10.7: 規範でも、補正でも、before/after でもない。

    関数のソースではなく利用者に見える文字列に対して検証する。禁止する枠付けを
    禁じるために名指しする docstring が、このチェックを落とさないようにする
    ためである。
    """
    shown = SYMMETRIZED_IMAGE_CAPTION + SYMMETRIZED_IMAGE_NOTICE + SYMMETRIZED_IMAGE_LIMITS

    for forbidden in ("理想", "正解", "補正後", "美化", "before", "after"):
        assert forbidden not in shown
    # そして、その画像が実際に何であるかを述べ続けなければならない。
    assert "加工画像" in shown


# --- §14.3e 品質ゲートの色 --------------------------------------------------


def test_every_gate_status_has_a_colour() -> None:
    """ゲートへ状態を追加したときに、KeyError へ落ちてはならない。"""
    for status in Status:
        assert gate_color(AnalysisState(status=status)) == GATE_COLORS[status]


def test_gate_colours_are_valid_hex() -> None:
    for colour in GATE_COLORS.values():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colour), colour


def test_a_missing_state_reads_as_the_neutral_colour() -> None:
    """待機中の表示が「条件を満たしている」色を借りてはならない。"""
    assert gate_color(None) == GATE_COLORS[Status.ERROR]


def test_the_blocking_states_are_visually_distinct_from_ok() -> None:
    """§5.4: 色は 1 つの伝達手段にすぎないが、それでも区別できなければならない。"""
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.HOLD]
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.NO_FACE]
    assert GATE_COLORS[Status.OK] != GATE_COLORS[Status.ERROR]
