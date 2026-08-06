import re

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.scores import (
    DISTANCE_UNIT_CAPTION,
    GATE_COLORS,
    SCORE_HEADING,
    SYMMETRIZED_IMAGE_CAPTION,
    SYMMETRIZED_IMAGE_LIMITS,
    SYMMETRIZED_IMAGE_NOTICE,
    format_distance,
    format_symmetry,
    gate_color,
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
