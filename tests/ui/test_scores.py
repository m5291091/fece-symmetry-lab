"""`ui.scores`: 詳細結果の描画と、そこに出る §5.3 の文言。

値の解決と整形は `test_readings.py`、対称化パネルは
`test_symmetrization_panel.py` が受け持つ。ここで見るのは「画面に何がどの順で
出るか」と「その言葉が §5.3 を守っているか」である。

**「全体のスコア」の計算過程開示は撤回した**（2026-09-03、§0 項目 34）。
"""

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import ReasonCode
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.readings import format_symmetry
from face_symmetry_lab.ui.scores import (
    DISTANCE_UNIT_CAPTION,
    PART_LEVEL_VARIATION_CAPTION,
    SCORE_HEADING,
)


def _run(body: str) -> AppTest:
    at = AppTest.from_string(
        "from face_symmetry_lab.domain.landmarks import TrackingStatus\n"
        "from face_symmetry_lab.ui.scores import render_tracking_status\n"
        "MESSAGES = {status: status.name.lower().replace('_', ' ') for status in TrackingStatus}\n"
        f"{body}\n"
    )
    at.run()
    return at


def _render_details(*, body: str) -> AppTest:
    """`render_score_details` を 1 回描画した `AppTest` を返す。"""
    at = AppTest.from_string(
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "from face_symmetry_lab.ui.scores import render_score_details\n" + body
    )
    at.run()
    assert not at.exception
    return at


def _code_text(at: AppTest) -> str:
    return " ".join(element.value for element in at.code)


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


def test_render_guidance_shows_warning_caption_for_low_light() -> None:
    """§6.2/§8.2: warning_codes=(LOW_LIGHT,) のとき REASON_GUIDANCE[LOW_LIGHT] が
    `st.caption` として出力される（阻害理由とは視覚的に区別）。"""
    at = AppTest.from_string(
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status\n"
        "from face_symmetry_lab.ui.scores import render_guidance\n"
        "from face_symmetry_lab.ui.guidance import REASON_GUIDANCE\n"
        "render_guidance(\n"
        "    AnalysisOutcome(\n"
        "        frame_id=1,\n"
        "        state=AnalysisState(\n"
        "            status=Status.OK,\n"
        "            warning_codes=(ReasonCode.LOW_LIGHT,),\n"
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

    assert not at.exception
    captions = " ".join(element.value for element in at.caption)
    assert REASON_GUIDANCE[ReasonCode.LOW_LIGHT] in captions


def test_render_guidance_reason_codes_still_appear_after_i18n_fix() -> None:
    """i18n 修正で回帰しないこと: reason_codes の文言は従来どおり出る。"""
    at = AppTest.from_string(
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status\n"
        "from face_symmetry_lab.ui.scores import render_guidance\n"
        "from face_symmetry_lab.ui.guidance import REASON_GUIDANCE\n"
        "render_guidance(\n"
        "    AnalysisOutcome(\n"
        "        frame_id=1,\n"
        "        state=AnalysisState(\n"
        "            status=Status.HOLD,\n"
        "            reason=ReasonCode.TOO_FAR,\n"
        "            reason_codes=(ReasonCode.TOO_FAR,),\n"
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

    assert not at.exception
    rendered = " ".join(element.value for element in at.markdown)
    assert REASON_GUIDANCE[ReasonCode.TOO_FAR] in rendered


def test_the_detail_panel_states_the_deviation_beside_the_symmetry() -> None:
    """§5.3/§8.6.3: 両方の向きを画面に出すので、100% が目標として通ることはない。

    見出し結果の削除（2026-08-12）でこの併記の受け皿は詳細パネルだけになった。
    罠: 併記ごと消した実装は、対称度が一方向からしか読めなくなってここで落ちる。
    """
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.0215\n"
            "    ),\n"
            "    smooth_3d=0.0215,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    captions = " ".join(element.value for element in at.caption)
    assert "2.15%" in captions
    assert "対称度との和は 100%" in captions


def test_the_symmetry_heading_and_caption_avoid_a_value_judgement() -> None:
    """§5.3.1（2026-08-16）: 常時表示は 1 行。根拠は `help` と About が持つ。"""
    from face_symmetry_lab.ui.scores import DISTANCE_UNIT_HELP

    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )
    combined = " ".join(str(element.value) for element in at.caption)

    assert "試験中の指標です" in combined
    assert DISTANCE_UNIT_CAPTION in combined
    assert "目標値ではありません" in DISTANCE_UNIT_CAPTION
    # 長い説明は `help` へ移した（常時は見えないが、その場で読める）。
    assert "狭い範囲に集まりやすく" in DISTANCE_UNIT_HELP
    assert "狭い範囲に集まりやすく" not in combined
    for banned in ("良い", "悪い", "美しい", "理想"):
        assert banned not in combined


# --- §8.6.3 部位別の内訳 ------------------------------------------------------


def test_the_part_breakdown_reads_from_the_three_dimensional_result() -> None:
    """§8.6.3: 部位別 4 つは `d_3d,p` から作る。2D の値（0.5）ではない。"""
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    values = {metric.label: metric.value for metric in at.metric}
    assert values["輪郭"] == format_symmetry(0.01)
    assert values["目"] == format_symmetry(0.02)
    assert values["鼻"] == format_symmetry(0.03)
    assert values["口"] == format_symmetry(0.015)


def test_the_part_breakdown_shows_the_smoothed_value_as_primary_and_notes_the_raw_one() -> None:
    """§8.7 改訂（H1）: 主表示は平滑値、詳細タブは生値を併記する。

    罠: 部位別だけ生値を主表示にする実装は、全体値と 2 つの表示規則を作る。
    """
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.011, eyes=0.022, nose=0.033, mouth=0.024, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            "    part_smooth_3d={\n"
            "        'contour': 0.014, 'eyes': 0.016, 'nose': 0.019, 'mouth': 0.015,\n"
            "    },\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=True)\n"
        )
    )

    values = {metric.label: metric.value for metric in at.metric}
    assert values["輪郭"] == format_symmetry(0.014)
    captions = " ".join(element.value for element in at.caption)
    assert f"生値 {format_symmetry(0.011)}" in captions


def test_a_single_missing_part_only_shows_that_parts_detail_reading_as_a_dash() -> None:
    """不変条件 11: 1 部位の欠測が他の 3 部位を巻き添えにしない。"""
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.01, eyes=0.02, nose=None, mouth=0.015, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    values = {metric.label: metric.value for metric in at.metric}
    assert values["鼻"] == "—"
    assert values["輪郭"] == format_symmetry(0.01)


def test_a_missing_three_dimensional_result_shows_every_part_reading_as_a_dash() -> None:
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(symmetry_3d=None, smooth_3d=None)\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    values = {metric.label: metric.value for metric in at.metric}
    assert [values[label] for label in ("輪郭", "目", "鼻", "口")] == ["—"] * 4


def test_the_part_level_variation_note_applies_equally_to_all_four_parts() -> None:
    """§5.3/OD-09: 特定の部位（鼻など）だけを名指しで「使えない」と述べない。"""
    for part in ("鼻", "目", "口", "輪郭"):
        assert part not in PART_LEVEL_VARIATION_CAPTION
    assert "部位ごとに異なり" in PART_LEVEL_VARIATION_CAPTION


def test_show_parts_false_hides_the_part_metrics_but_keeps_the_required_notices() -> None:
    """項目 26（H3）: Camera は HUD が部位別を出すため、詳細タブでは省く。

    罠: 部位別セクションを丸ごと削る実装だと、§5.3 が義務付ける見出しと注記
    まで一緒に消えてここで落ちる。
    """
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False, show_parts=False)\n"
        )
    )

    labels = [metric.label for metric in at.metric]
    for part in ("輪郭", "目", "鼻", "口"):
        assert part not in labels

    captions = " ".join(element.value for element in at.caption)
    # §5.3.1（2026-08-16）: 部位別の揺れ幅は `help` へ移した。部位別を出さない
    # ときは、その注記も要らない。
    assert PART_LEVEL_VARIATION_CAPTION not in captions
    assert SCORE_HEADING in [element.value for element in at.subheader]
    # §5.3.1: `st.info` の 2 文から `st.caption` の 1 行へ。「試験中の指標」で
    # あること（不変条件 8）は残す。
    assert any("試験中の指標" in element.value for element in at.caption)


def test_show_parts_false_still_shows_pose_and_metric_version_captions() -> None:
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False, show_parts=False)\n"
        )
    )

    captions = " ".join(element.value for element in at.caption)
    assert "姿勢" in captions
    assert "symmetry-3d/test" in captions


def test_render_score_details_show_parts_defaults_to_true() -> None:
    """Image はこの既定値のまま呼ぶ。既定を `False` に変えると Image 側からも
    部位別が消えるため、パラメータの既定値そのものを固定する。"""
    import inspect

    from face_symmetry_lab.ui.scores import render_score_details

    signature = inspect.signature(render_score_details)

    assert signature.parameters["show_parts"].default is True


def test_the_closing_limits_are_a_single_line() -> None:
    """§5.3.1（2026-08-16）: 見出し付きの箱に 3 本の長文を並べない。

    罠: 以前の版はここに 3 本、その上にさらに 2 本の caption があり、灰色の壁だった。
    **読まれない注意文は、書いていないのと同じである。**
    """
    from face_symmetry_lab.ui.scores import READING_LIMITS

    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )
    captions = [str(element.value) for element in at.caption]

    assert len(READING_LIMITS) == 1
    assert len(READING_LIMITS[0]) < 100
    assert any(READING_LIMITS[0] in caption for caption in captions)


def test_the_closing_limit_still_states_what_it_cannot_be_used_for() -> None:
    """減らしたのは分量であって、禁止そのものではない（§5.3.1）。"""
    from face_symmetry_lab.ui.scores import READING_LIMITS

    text = " ".join(READING_LIMITS)

    assert "他人との比較" in text
    assert "医学的評価" in text
    for banned in ("良い", "悪い", "美しい", "理想", "偏差値", "順位"):
        assert banned not in text


def test_the_metric_versions_stay_separate_from_the_caveats() -> None:
    """版と対応表は来歴であって留保ではない。同じ箱へ混ぜない（FR-020）。"""
    from face_symmetry_lab.ui.scores import READING_LIMITS

    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    version_caption = next(
        element.value for element in at.caption if "landmark map" in element.value
    )

    assert "symmetry-3d/test" in version_caption
    assert all(limit not in version_caption for limit in READING_LIMITS)


# --- §0 項目 34（撤回、2026-09-03）: 「全体のスコア」の計算過程開示 -------------


def test_the_detail_panel_has_no_calculation_walkthrough_but_keeps_the_rest() -> None:
    """§8.6.7 の表示スケールと H4 の 3 ステップ開示を撤回した。

    罠: 部位別まで巻き添えで消す実装は、この test の後半（部位別 4 つ・
    SCORE_HEADING・「試験中の指標」・部位別の揺れ幅の注記・姿勢）で落ちる。
    """
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    # 計算過程の expander が無い。
    assert not at.expander
    assert not at.code
    combined = " ".join(str(element.value) for element in (*at.markdown, *at.caption, *at.info))
    for banned in ("計算過程", "ステップ"):
        assert banned not in combined

    # 部位別 4 つ・見出し・「試験中の指標」・揺れ幅の注記・姿勢は従来どおり出る。
    labels = [metric.label for metric in at.metric]
    for part in ("輪郭", "目", "鼻", "口"):
        assert part in labels
    assert SCORE_HEADING in [element.value for element in at.subheader]
    assert any("試験中の指標" in element.value for element in at.caption)
    assert "部位別の対称度" in combined
    assert any("姿勢" in element.value for element in at.caption)
