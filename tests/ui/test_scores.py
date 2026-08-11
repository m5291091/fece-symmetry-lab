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
    resolve_overall_distance,
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
) -> AnalysisOutcome:
    """§8.6.3 の表示テスト用 `AnalysisOutcome`。

    2D の値（既定 0.5）は 3D の値（テストでは概ね 0.0〜0.1）から大きく離す。
    表示ロジックが誤って 2D を読んでいれば、この差で assertion が落ちる
    （fixture 設計上の罠、いちばん重要な部分）。
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
    at = _run("render_headline_metrics(overall=None, estimate=None)")

    assert not at.exception
    assert [element.value for element in at.metric] == ["—", "—", "—"]


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
        "render_headline_metrics(overall=0.0215, estimate=None)\n"
    )
    at.run()

    assert not at.exception
    assert at.metric[0].label == "全体の対称度"
    assert at.metric[0].value == "97.85%"
    captions = " ".join(element.value for element in at.caption)
    assert "2.15%" in captions


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


def test_the_part_equal_weight_symmetry_is_a_distinct_reading_from_the_headline_overall() -> None:
    """§8.6.5: `s_3d`（部位平均）は `d_3d`（全体）と重みだけが違う別の総合量で、
    一致しない。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import (\n"
        "    render_headline_metrics,\n"
        "    render_score_details,\n"
        "    resolve_overall_distance,\n"
        ")\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.005, eyes=0.005, nose=0.06, mouth=0.005, overall=0.02\n"
        "    ),\n"
        "    smooth_3d=0.02,\n"
        ")\n"
        "render_headline_metrics(overall=resolve_overall_distance(outcome), estimate=None)\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    assert values["全体の対称度"] == format_symmetry(0.02)
    assert values[PART_EQUAL_WEIGHT_LABEL] != values["全体の対称度"]


def test_a_single_missing_part_only_removes_the_part_equal_weight_reading() -> None:
    """§8.6.5: 部位が 1 つ欠測しても `s_3d` だけが `—` になり、全体・他部位は
    出続ける（AC-SCORE-10 と同じ非対称な欠測条件）。"""
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
    assert values[PART_EQUAL_WEIGHT_LABEL] == "—"


def test_a_missing_three_dimensional_result_shows_every_reading_as_a_dash() -> None:
    """不変条件 11: §8.5.1 の重ね合わせが成立しないとき、0 や前値ではなく `—`。"""
    at = AppTest.from_string(
        "from tests.ui.test_scores import _outcome_for_display\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n"
        "outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)\n"
        "render_score_details(outcome, show_smoothed=False)\n"
    )
    at.run()

    assert not at.exception
    values = {metric.label: metric.value for metric in at.metric}
    for label in ("輪郭", "目", "鼻", "口", PART_EQUAL_WEIGHT_LABEL):
        assert values[label] == "—"


def test_the_part_level_variation_note_applies_equally_to_all_four_parts() -> None:
    """§5.3: 撮影のたびに動く幅より小さい変化は読み取れないという注記を、特定の
    部位（鼻など）だけを名指しして述べない。"""
    for singled_out in ("鼻だけ", "鼻のみ", "鼻は使えません", "鼻は信頼できません"):
        assert singled_out not in PART_LEVEL_VARIATION_CAPTION
    assert "撮影" in PART_LEVEL_VARIATION_CAPTION


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
