"""`ui.scores`: 詳細結果の描画と、そこに出る §5.3 の文言。

値の解決と整形は `test_readings.py`、対称化パネルは
`test_symmetrization_panel.py`、計算過程の算術は
`tests/domain/test_score_explanation.py` が受け持つ。ここで見るのは
「画面に何がどの順で出るか」と「その言葉が §5.3 を守っているか」である。
"""

import unicodedata

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.models import ReasonCode
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE
from face_symmetry_lab.ui.readings import format_distance, format_symmetry
from face_symmetry_lab.ui.scores import (
    DISTANCE_UNIT_CAPTION,
    PART_EQUAL_WEIGHT_HELP,
    PART_EQUAL_WEIGHT_LABEL,
    PART_LEVEL_VARIATION_CAPTION,
    SCORE_HEADING,
    SCORE_STEP_TITLES,
)
from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result


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


# --- §5.3 の文言 --------------------------------------------------------------


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


def test_the_detail_panel_carries_the_score_help_that_the_headline_used_to_hold() -> None:
    """§8.6.3: 「採点ではない」「順位ではない」「アンカー 2 点」は必ず画面に出す。

    見出し結果の削除（2026-08-12）で `help=` の置き場が無くなったため、計算過程の
    節がこの説明の唯一の受け皿になる。罠: 見出しと一緒に説明ごと消した実装は、
    引き伸ばした点数だけが何の断りもなく残ってここで落ちる。
    """
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

    assert PART_EQUAL_WEIGHT_HELP in [element.value for element in at.caption]


def test_the_part_equal_weight_headline_label_uses_the_user_specified_wording() -> None:
    """ユーザー指定により「部位平均の対称度」ではなく「全体のスコア」を見出しの
    ラベルにする（2026-08-11、W5-fix）。"""
    assert PART_EQUAL_WEIGHT_LABEL == "全体のスコア"


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


def test_the_detail_panel_no_longer_duplicates_the_headline_score() -> None:
    """項目 23: 「全体のスコア」相当の表示は吹き出しと HUD の側にしかない。

    罠: 詳細タブへメトリクスとして再掲する実装は、同じ測定値を 2 か所へ出す。
    """
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

    labels = [metric.label for metric in at.metric]
    assert PART_EQUAL_WEIGHT_LABEL not in labels


def test_the_part_level_variation_note_applies_equally_to_all_four_parts() -> None:
    """§5.3/OD-09: 特定の部位（鼻など）だけを名指しで「使えない」と述べない。"""
    for part in ("鼻", "目", "口", "輪郭"):
        assert part not in PART_LEVEL_VARIATION_CAPTION
    assert "部位ごとに異なり" in PART_LEVEL_VARIATION_CAPTION


def test_show_parts_false_hides_the_part_metrics_but_keeps_the_required_notices() -> None:
    """項目 26（H3）: Camera は HUD が部位別を出すため、詳細タブでは省く。

    罠: 部位別セクションを丸ごと削る実装だと、§5.3 が義務付ける注記と計算過程
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
    assert PART_LEVEL_VARIATION_CAPTION in captions
    assert SCORE_HEADING in [element.value for element in at.subheader]
    assert any("試験中の指標" in element.value for element in at.info)
    # 計算過程は残り続ける（部位別 4 つの値も、その中には出る）。
    walkthrough = _code_text(at)
    for deviation in ("2.11", "1.44", "0.62", "1.83"):
        assert deviation in walkthrough


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


# --- §8.6.3(H4): 「全体のスコア」の計算過程を 3 ステップで開示する -------------


#: 3 ステップの検証で使い回す代表値。`s_3d` = 1.60%、点 = 85.0 点になる。
#: どの値も §8.6.7 のアンカー表記（1.00 / 5.00）と衝突しない。
_STEP_SAMPLE = (
    "outcome = _outcome_for_display(\n"
    "    symmetry_3d=_symmetry_3d_result(\n"
    "        contour=0.0211, eyes=0.0144, nose=0.0062, mouth=0.0183, overall=0.018\n"
    "    ),\n"
    "    smooth_3d=0.018,\n"
    ")\n"
    "render_score_details(outcome, show_smoothed=False)\n"
)


def test_the_walkthrough_presents_three_numbered_steps_in_order() -> None:
    """H4: 式を 1 つの塊で見せず、手順として順番に並べる。

    罠: 見出しだけ足して中身が 1 ブロックのままの実装は、順序どおりに 3 つの
    見出しが並ぶというこの assertion では通ってしまうが、続く 3 件
    （各ステップの中身）が別々の `st.code` を要求するため落ちる。
    """
    at = _render_details(body=_STEP_SAMPLE)

    rendered = [element.value for element in at.markdown]
    positions = [rendered.index(f"**{title}**") for title in SCORE_STEP_TITLES]
    assert positions == sorted(positions)


def test_step_one_lists_each_part_with_its_deviation_and_square() -> None:
    """ステップ 1: 4 部位のずれと、次で使う二乗を 1 行ずつ並べる。"""
    at = _render_details(body=_STEP_SAMPLE)

    blocks = [element.value for element in at.code]
    step_one = next(block for block in blocks if "ずれ²" in block and "輪郭" in block)
    for label in ("輪郭", "目", "鼻", "口"):
        assert label in step_one
    for deviation in ("2.11%", "1.44%", "0.62%", "1.83%"):
        assert deviation in step_one
    # 二乗（パーセントポイントの二乗）。2.11² = 4.4521 → 4.45
    assert "4.45" in step_one


def _cells(text: str) -> int:
    """等幅フォントで占める桁数。**production の実装を借りない。**

    `ui.scores._display_width` を使うと、その関数が壊れたときに測る側も一緒に
    壊れ、桁ずれを検出できないテストになる（実際にそうなっていた）。ここでは
    独立に数える。
    """
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def test_step_one_aligns_its_columns_for_a_monospace_font() -> None:
    """H4: 表として縦に目で追えなければ、ステップ 1 は表の意味を持たない。

    罠: `str.ljust` や `f"{x:>8}"` は **文字数** で詰めるため、「輪郭」（2 文字
    = 4 桁）と「目」（1 文字 = 2 桁）で桁がずれる。全角を 2 桁として数える実装
    でなければここで落ちる。
    """
    at = _render_details(body=_STEP_SAMPLE)

    blocks = [element.value for element in at.code]
    step_one = next(block for block in blocks if "ずれ²" in block and "輪郭" in block)
    rows = step_one.splitlines()

    assert len(rows) == 5  # 見出し + 4 部位
    widths = {_cells(row) for row in rows}
    assert len(widths) == 1, f"桁が揃っていない: {[(_cells(row), row) for row in rows]}"


def test_step_three_aligns_its_continuation_lines_under_the_first_equals() -> None:
    """H4: 続く 2 行の `=` が 1 行目の `=` の真下に来る。

    罠: ラベル「全体のスコア」は 6 文字 = 12 桁なので、`len()` で詰めた実装は
    `=` が 6 桁ぶん左へずれ、式が階段状に崩れる。
    """
    at = _render_details(body=_STEP_SAMPLE)

    blocks = [element.value for element in at.code]
    step_three = next(block for block in blocks if PART_EQUAL_WEIGHT_LABEL in block)
    columns = {_cells(row.split("=")[0]) for row in step_three.splitlines()}

    assert len(columns) == 1, f"`=` の位置が揃っていない: {step_three}"


def test_step_two_shows_the_sum_the_mean_and_the_square_root() -> None:
    """ステップ 2: 途中経過を出す。**読者が電卓で追える形にする。**

    罠: 最終値だけを出して途中を飛ばす実装（旧 H2 の 1 行式）はここで落ちる。
    """
    at = _render_details(body=_STEP_SAMPLE)

    blocks = [element.value for element in at.code]
    step_two = next(block for block in blocks if "ずれ² の合計" in block)
    assert "4 部位で平均" in step_two
    assert "平方根をとる" in step_two
    assert "1.60%" in step_two  # sqrt(mean(2.11²+1.44²+0.62²+1.83²))


def test_step_three_names_both_anchors_and_converts_into_points() -> None:
    """§8.6.7: アンカー 2 点を明示したうえで点へ換算する。

    罠: `100 - ずれ` のままの実装は `98.40` を出して落ちる。
    """
    at = _render_details(body=_STEP_SAMPLE)

    blocks = [element.value for element in at.code]
    step_three = next(block for block in blocks if PART_EQUAL_WEIGHT_LABEL in block)
    for anchor in ("1.00", "5.00"):
        assert anchor in step_three
    assert "85.0 点" in step_three
    assert "98.40" not in _code_text(at)


def test_the_walkthrough_states_the_slope_from_the_frozen_anchors() -> None:
    """§8.6.7: ずれが 1 ポイント動くと点は 25 点動く。

    罠: UI 側へ `25` を直書きした実装は、アンカーを差し替えた版で嘘になる。
    ここでは `domain.score_scale` の値から組み立てられていることを固定する。
    """
    from face_symmetry_lab.domain.score_scale import POINTS_PER_DEVIATION_PERCENT
    from face_symmetry_lab.ui.scores import SCORE_CALCULATION_SLOPE_NOTE

    assert f"{POINTS_PER_DEVIATION_PERCENT:.0f}点動きます" in SCORE_CALCULATION_SLOPE_NOTE

    at = _render_details(body=_STEP_SAMPLE)
    assert SCORE_CALCULATION_SLOPE_NOTE in [element.value for element in at.caption]


def test_step_two_explains_why_the_mean_is_taken_over_squares() -> None:
    """H4: 「二乗して平方根」が儀式に見えないよう、理由を平叙文で述べる。

    ここが無いと初学者は最初のステップで脱落する。
    """
    from face_symmetry_lab.ui.scores import SCORE_CALCULATION_COMBINE_INTRO

    assert "薄められて" in SCORE_CALCULATION_COMBINE_INTRO
    assert "二乗平均平方根" in SCORE_CALCULATION_COMBINE_INTRO


def test_step_one_gives_the_deviation_a_physical_size() -> None:
    """H4: 「両眼間隔の 1%」だけでは大きさが掴めない。実寸の目安を添える。"""
    from face_symmetry_lab.ui.scores import SCORE_CALCULATION_DEVIATION_INTRO

    assert "0.6mm" in SCORE_CALCULATION_DEVIATION_INTRO


def test_the_walkthrough_matches_the_part_level_metrics_above_it() -> None:
    """H2/H4: 計算過程の部位別の値は、上の部位別メトリクス（主表示、H1 の平滑値）と
    文字列として一致する。罠 (b): 計算過程だけ生値（1.10/2.20/3.30/2.40）を使うと
    ここが落ちる。

    生値・平滑値のいずれも §8.6.7 のアンカー表記（`1.00` / `5.00`）と衝突しない
    値を選んでいる。衝突すると、生値が出ていないことをこの assertion で示せない。
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

    metric_values = {metric.label: metric.value for metric in at.metric}
    walkthrough = _code_text(at)
    for label, smoothed in (("輪郭", 0.014), ("目", 0.016), ("鼻", 0.019), ("口", 0.015)):
        assert metric_values[label] == format_symmetry(smoothed)
        assert format_distance(smoothed) in walkthrough
    for raw_deviation in ("1.10", "2.20", "3.30", "2.40"):
        assert raw_deviation not in walkthrough


def test_the_walkthrough_matches_resolve_part_distance_when_the_metrics_are_hidden() -> None:
    """AC-DISPLAY-04: Camera（`show_parts=False`）では部位別メトリクスが描かれ
    ないため、上の表示とは比較できない。代わりに、計算過程が使う値が
    `resolve_part_distance_3d`（HUD が `format_symmetry` を通す前と同じ値）から
    直接一致することを確認する。罠: 計算過程だけ別の値（例えば生値）を使う実装
    はこの assertion が落ちる。"""
    from face_symmetry_lab.ui.readings import resolve_part_distance_3d

    at = _render_details(
        body=(
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
    )

    assert not at.metric  # HUD が既に部位別を表示しているため、ここには無い

    outcome = _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.018,
        part_smooth_3d={"contour": 0.004, "eyes": 0.006, "nose": 0.009, "mouth": 0.005},
    )
    walkthrough = _code_text(at)
    for name in ("contour", "eyes", "nose", "mouth"):
        value = resolve_part_distance_3d(outcome, name)
        assert format_distance(value) in walkthrough


def test_the_walkthrough_shows_no_numbers_when_a_part_is_missing() -> None:
    """H4/不変条件 11: `s_3d` が部位欠測で欠測のとき、計算過程は数値を代入しない
    （罠: 欠測を 0 へ fallback すると `100.0 点` という偽の上限が出てしまう）。

    §8.6.7 のアンカー（`1.00%` / `5.00%`）は式の形の一部なので、値が無くても
    出続ける。輪郭の生値はそれと衝突しない `1.10%` にしてある。
    """
    at = _render_details(
        body=(
            "outcome = _outcome_for_display(\n"
            "    symmetry_3d=_symmetry_3d_result(\n"
            "        contour=0.011, eyes=0.022, nose=None, mouth=0.015, overall=0.018\n"
            "    ),\n"
            "    smooth_3d=0.018,\n"
            ")\n"
            "render_score_details(outcome, show_smoothed=False)\n"
        )
    )

    walkthrough = _code_text(at)
    assert "100.0 点" not in walkthrough
    assert "1.10" not in walkthrough  # 輪郭のずれを代入した行そのものが無い
    assert "ずれ² の合計" not in walkthrough  # 途中経過も出ない
    # 式の形とアンカーは残る（何を計算しているかは説明し続ける）。
    assert "1.00" in walkthrough and "5.00" in walkthrough
    notices = " ".join(element.value for element in list(at.info) + list(at.caption))
    assert "当てはめられません" in notices


def test_the_walkthrough_keeps_the_caveats_after_the_steps_rather_than_inside_them() -> None:
    """H4: 手順と、手順の読み方に対する留保は別のものである。

    4 つの注記（重みの違い・目盛りを広げた理由・実測レンジ・help）は 3 つの
    ステップがすべて終わったあとに置く。罠: ステップの間へ挟む実装は、手順を
    追っている途中で読者を注意書きへ迷い込ませる。
    """
    from face_symmetry_lab.ui.scores import (
        SCORE_CALCULATION_HIGH_VALUE_NOTE,
        SCORE_CALCULATION_MEASURED_RANGE_CAPTION,
        SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION,
    )

    at = _render_details(body=_STEP_SAMPLE)

    captions = [element.value for element in at.caption]
    markdown = [element.value for element in at.markdown]
    last_step_caption_index = captions.index(
        next(caption for caption in captions if "1ポイント" in caption)
    )
    for note in (
        PART_EQUAL_WEIGHT_HELP,
        SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION,
        SCORE_CALCULATION_HIGH_VALUE_NOTE,
        SCORE_CALCULATION_MEASURED_RANGE_CAPTION,
    ):
        assert captions.index(note) > last_step_caption_index
    assert f"**{SCORE_STEP_TITLES[2]}**" in markdown


def test_the_walkthrough_uses_no_progress_bar_or_gauge() -> None:
    """§5.2: 「達成度」として演出しない。目盛りの端点は数値と文章で示す。

    罠: 点数の位置をバーで見せる実装は、§8.6.7 のアンカーが母集団の分位で
    あるかのように読ませる。
    """
    import inspect

    from face_symmetry_lab.ui import scores

    source = inspect.getsource(scores)
    for widget in ("st.progress", "st.slider", "progress_bar"):
        assert widget not in source


def test_the_walkthrough_wording_avoids_value_judgements_and_thresholds() -> None:
    """§5.3/§10.7/不変条件 8: 計算過程の説明文にも他の文言と同じ禁止語が適用され、
    加えて母集団の分布を持たないため基準値も書かない。"""
    from face_symmetry_lab.ui.scores import (
        SCORE_CALCULATION_COMBINE_INTRO,
        SCORE_CALCULATION_DEVIATION_INTRO,
        SCORE_CALCULATION_HIGH_VALUE_NOTE,
        SCORE_CALCULATION_MEASURED_RANGE_CAPTION,
        SCORE_CALCULATION_SCALE_INTRO,
        SCORE_CALCULATION_SLOPE_NOTE,
        SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION,
    )

    combined = "".join(
        (
            *SCORE_STEP_TITLES,
            SCORE_CALCULATION_DEVIATION_INTRO,
            SCORE_CALCULATION_COMBINE_INTRO,
            SCORE_CALCULATION_SCALE_INTRO,
            SCORE_CALCULATION_SLOPE_NOTE,
            SCORE_CALCULATION_WEIGHT_DIFFERENCE_CAPTION,
            SCORE_CALCULATION_HIGH_VALUE_NOTE,
            SCORE_CALCULATION_MEASURED_RANGE_CAPTION,
        )
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
        "理想",
    ):
        assert forbidden not in combined
    for threshold_phrase in ("以上なら対称", "%以上で対称", "以上あれば対称"):
        assert threshold_phrase not in combined


def test_the_walkthrough_notes_the_single_subject_caveat() -> None:
    """OD-09: 個人差は未検証であるため「被写体 1 名」の但し書きを必ず書く。"""
    from face_symmetry_lab.ui.scores import SCORE_CALCULATION_MEASURED_RANGE_CAPTION

    assert "被写体 1 名" in SCORE_CALCULATION_MEASURED_RANGE_CAPTION


def test_the_score_help_states_the_anchors_and_denies_a_population_rank() -> None:
    """§8.6.7/AC-DISPLAY-05: help はアンカー 2 点を述べ、順位ではないことを明示する。

    アンカーが画面に無いと、§8.6.3 が要求する可逆性（`s_3d` へ戻せること）が
    利用者側で成立しない。
    """
    assert "100 点" in PART_EQUAL_WEIGHT_HELP
    assert "0 点" in PART_EQUAL_WEIGHT_HELP
    assert "1.0%" in PART_EQUAL_WEIGHT_HELP
    assert "5.0%" in PART_EQUAL_WEIGHT_HELP
    assert "順位" in PART_EQUAL_WEIGHT_HELP
    assert "採点" in PART_EQUAL_WEIGHT_HELP


def test_the_score_scale_caption_warns_that_a_reshoot_moves_the_points() -> None:
    """§8.6.7: 目盛りを 25 倍にすると、ばらつきも同じ 25 倍で拡大する。

    実測（§14.4 セッション間全幅 0.273pt = 6.8 点）を利用者へ開示し、その幅より
    小さい点差を人物間の差として読ませない。
    """
    from face_symmetry_lab.ui.scores import SCORE_SCALE_CAPTION

    assert "撮り直" in SCORE_SCALE_CAPTION
    assert "6.8" in SCORE_SCALE_CAPTION
    for forbidden in ("良い", "悪い", "美しい", "優秀", "優れ", "満点", "合格"):
        assert forbidden not in SCORE_SCALE_CAPTION


def test_the_score_scale_caption_is_rendered_outside_the_expander() -> None:
    """§8.6.7: 注記は定数として存在するだけでなく、畳まれていない位置に出る。"""
    from face_symmetry_lab.ui.scores import SCORE_SCALE_CAPTION

    at = _render_details(body=_STEP_SAMPLE)

    assert SCORE_SCALE_CAPTION in [element.value for element in at.caption]
