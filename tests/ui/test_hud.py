"""§14.3e の HUD 資産に対するテスト。

ここで検証するのはブラウザへ渡る文字列そのもの（CSS・実行時 JS・5 Hz の
payload）である。HUD をカメラ画面の合成から切り離したので、テストも同じ境界
で分ける。カメラ側の配線（どの fragment が payload を publish するか等）は
`test_camera.py` が受け持つ。
"""

import json

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.age_gate import AgeGateStatus
from face_symmetry_lab.application.reference_baseline import BaselineSummary
from face_symmetry_lab.domain.models import AnalysisState, GenderLabel, Status
from face_symmetry_lab.domain.reference_stats import MetricSummary
from face_symmetry_lab.ui.age_gate_notice import age_gate_notice
from face_symmetry_lab.ui.hud import (
    HUD_CHART_CAPACITY,
    HUD_CHART_WINDOW_SECONDS,
    HUD_CHART_Y_MAX,
    HUD_CHART_Y_MIN,
    HUD_PLACEMENTS,
    HUD_POSITION_OPTIONS,
    HUD_REFRESH_MS,
    HUD_RUNTIME_JS,
    HUD_STYLESHEET,
    build_hud_payload,
    hud_bridge_script,
    hud_html,
)
from face_symmetry_lab.ui.readings import (
    GATE_LABELS,
    GENDER_LABELS,
    PART_LABELS,
    format_score,
    format_smoothing_window,
    format_symmetry,
    gate_label,
)


def _estimate(age: float = 34.0, female: float = 0.8):
    from datetime import UTC, datetime

    from face_symmetry_lab.domain.demographics import build_demographic_estimate

    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(female, 1.0 - female),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def _payload_kwargs() -> dict:
    return {
        "overall": 0.0215,
        "equal_weight": 0.031,
        "parts": {"contour": 0.01, "eyes": 0.02, "nose": 0.03, "mouth": 0.015},
        "estimate": None,
        "show_demographics": False,
        "gate": "#2E9E5B",
        "position": "左上",
        "generation": "segment-1",
    }


def _payload(**overrides):
    settings = _payload_kwargs()
    settings.update(overrides)
    return build_hud_payload(**settings)


def _bridge(**overrides) -> str:
    settings = {"fullscreen": False, "position": "左上"}
    settings.update(overrides)
    return hud_bridge_script(**settings)


# --- payload の値 -------------------------------------------------------------


def test_the_hud_reuses_the_shared_symmetry_formatter() -> None:
    """同じ測定値に対する 2 つ目の表示規則を作らない（§14.3e）。"""
    assert _payload()["symmetry"] == format_symmetry(0.0215)


def test_the_hud_payload_reads_the_three_dimensional_overall_distance() -> None:
    """§8.6.3/§8.7: HUD の全体値も 3D の平滑値から来る（2D の値ではないことを固定）。

    `render_hud_payload`（snapshot からの配線）を対象にする。`build_hud_payload`
    は既に `overall` を受け取るだけの純関数として上のテストで確認済みなので、
    ここで確かめるのは「どの値を渡すか」という解決の部分である。
    """
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        "    symmetry_2d_overall=0.5,\n"
        ")\n"
        "render_hud_payload(PipelineSnapshot(analysis=outcome), show_demographics=False)\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["symmetry"] == format_symmetry(0.021)
    assert payload["symmetry"] != format_symmetry(0.5)


def test_a_missing_symmetry_value_renders_as_a_dash() -> None:
    """NFR-013: 欠測は欠測として示し、古い値や 0 を出さない。"""
    assert _payload(overall=None)["symmetry"] == "—"
    assert "—" in hud_html(**{**_payload_kwargs(), "overall": None})


def test_the_hud_carries_the_experimental_label() -> None:
    assert _payload()["label"] == "対称度（実験値）"


# --- H1: HUD をライブ解析パネルの内容へ拡張する ------------------------------


def test_the_hud_carries_the_headline_score_through_the_shared_formatter() -> None:
    """§8.6.5/§8.6.7: 「全体のスコア」（`s_3d`）も HUD へ、パネルと同じフォーマッタ
    で届く。**そのフォーマッタは点の `format_score` であり、%の `format_symmetry`
    ではない**（罠: HUD だけ対称度%のままにすると見出しと 2 つの目盛りが並ぶ）。"""
    assert _payload(equal_weight=0.031)["score"] == format_score(0.031)
    assert _payload(equal_weight=0.031)["score"] != format_score(0.0215)
    assert _payload(equal_weight=0.031)["score"] != format_symmetry(0.031)


def test_the_hud_carries_all_four_parts_through_the_shared_formatter() -> None:
    """§8.6.3: 4 部位も `ui.scores.format_symmetry` を通す（HUD 独自の整形をしない）。"""
    parts = _payload()["parts"]

    assert parts["contour"] == format_symmetry(0.01)
    assert parts["eyes"] == format_symmetry(0.02)
    assert parts["nose"] == format_symmetry(0.03)
    assert parts["mouth"] == format_symmetry(0.015)


def test_a_missing_score_or_part_renders_as_a_dash_not_zero() -> None:
    """NFR-013: 欠測は `—` であり、0 や前回値ではない。"""
    payload = _payload(
        equal_weight=None,
        parts={"contour": None, "eyes": 0.02, "nose": None, "mouth": 0.015},
    )

    assert payload["score"] == "—"
    assert payload["parts"]["contour"] == "—"
    assert payload["parts"]["nose"] == "—"
    assert payload["parts"]["eyes"] == format_symmetry(0.02)


def test_every_part_label_used_by_the_panel_has_a_hud_reading() -> None:
    """HUD の部位別行は `ui.scores.PART_LABELS` と同じ 4 部位を持つ。"""
    assert set(_payload()["parts"]) == set(PART_LABELS)


def test_the_hud_avoids_any_value_judgement() -> None:
    """§5.3: 数値を映像へ寄せても、HUD が評価になってはならない。"""
    rendered = repr(_payload(estimate=_estimate(), show_demographics=True))

    for forbidden in ("良い", "悪い", "改善", "スコアが高い", "優秀", "理想"):
        assert forbidden not in rendered


def test_the_hud_shows_age_and_gender_through_the_shared_labels() -> None:
    demo = _payload(estimate=_estimate(age=34.4), show_demographics=True)["demo"]

    assert "参考年齢 34 歳" in demo
    assert GENDER_LABELS[GenderLabel.FEMALE] in demo


def test_the_demographic_row_is_dropped_rather_than_dashed_when_absent() -> None:
    """FR-024: 空行は「機能が動いていて失敗している」という含意になる。"""
    payload = _payload(estimate=None, show_demographics=True)

    assert payload["demo"] == ""
    assert payload["symmetry"] == format_symmetry(0.0215)


def test_the_demographic_row_is_dropped_when_the_user_turned_it_off() -> None:
    assert _payload(estimate=_estimate(), show_demographics=False)["demo"] == ""


def test_the_hud_carries_the_gate_colour() -> None:
    assert _payload(gate="#D9A21B")["gate"] == "#D9A21B"


# --- FR-038: 年齢ゲートの通知 --------------------------------------------------


def test_a_notice_replaces_the_demographic_row_when_demographics_are_hidden() -> None:
    """FR-038: `show_demographics=False` のとき、`notice` が `demo` の代わりに入る。"""
    payload = _payload(estimate=None, show_demographics=False, notice="この画像では…")

    assert payload["demo"] == "この画像では…"


def test_a_notice_is_ignored_while_demographics_are_shown() -> None:
    """罠: `notice` を優先すると、通った判定の HUD にも余計な文言が混ざる。"""
    payload = _payload(
        estimate=_estimate(age=34.4), show_demographics=True, notice="出てはいけない"
    )

    assert "出てはいけない" not in payload["demo"]
    assert "参考年齢 34 歳" in payload["demo"]


def test_render_hud_payload_hides_demographics_and_shows_the_notice_when_the_gate_blocks() -> None:
    """FR-038: 年齢ゲートが参考年齢を止めているとき、HUD は数値ではなく通知を出す。"""
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "snapshot = PipelineSnapshot(\n"
        "    analysis=outcome,\n"
        "    age_gate_outcome=AgeGateOutcome(\n"
        "        status=AgeGateStatus.OUT_OF_RANGE, decision=None, reason=None\n"
        "    ),\n"
        ")\n"
        "render_hud_payload(snapshot, show_demographics=True)\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["demo"] == age_gate_notice(AgeGateStatus.OUT_OF_RANGE)
    assert payload["demo"] != ""


def test_render_hud_payload_ignores_a_gate_outcome_when_the_user_turned_demographics_off() -> None:
    """トグルを切っているだけのときは、ゲート通知を出さず沈黙のままにする。"""
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "snapshot = PipelineSnapshot(\n"
        "    analysis=outcome,\n"
        "    age_gate_outcome=AgeGateOutcome(\n"
        "        status=AgeGateStatus.OUT_OF_RANGE, decision=None, reason=None\n"
        "    ),\n"
        ")\n"
        "render_hud_payload(snapshot, show_demographics=False)\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["demo"] == ""


def test_the_payload_carries_a_generation_identifier_that_changes_with_the_segment() -> None:
    """§14.3e (H1): 系列の世代識別子。`segment_id` が変わると値も変わる。"""
    first = _payload(generation="segment-1")["gen"]
    second = _payload(generation="segment-2")["gen"]

    assert first == "segment-1"
    assert second == "segment-2"
    assert first != second


def test_the_hud_payload_reads_the_recorders_segment_id_as_the_generation() -> None:
    """`render_hud_payload`（snapshot からの配線）が `PipelineSnapshot.segment_id`
    を世代識別子として使うことを確かめる（`build_hud_payload` 自体は上のテストで
    純関数として確認済み）。"""
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "render_hud_payload(\n"
        "    PipelineSnapshot(analysis=outcome, segment_id='segment-xyz'),\n"
        "    show_demographics=False,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["gen"] == "segment-xyz"


def test_the_hud_payload_reads_the_equal_weight_score_and_parts_from_the_outcome() -> None:
    """`render_hud_payload` が `resolve_equal_weight_distance`/
    `resolve_part_distance_3d` を経由し、パネルと同じ値を読むことを固定する。

    部位別の平滑値（0.004 等）を生値（0.01 等）から大きく離しているため、
    実装が誤って生値をそのまま payload へ渡していればここで assertion が
    落ちる（fixture 設計上の罠）。
    """
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        "    part_smooth_3d={\n"
        "        'contour': 0.004, 'eyes': 0.006, 'nose': 0.009, 'mouth': 0.005,\n"
        "    },\n"
        ")\n"
        "render_hud_payload(PipelineSnapshot(analysis=outcome), show_demographics=False)\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])

    assert payload["score"] is not None
    assert payload["parts"]["contour"] == format_symmetry(0.004)
    assert payload["parts"]["contour"] != format_symmetry(0.01)
    assert payload["parts"]["eyes"] == format_symmetry(0.006)
    assert payload["parts"]["nose"] == format_symmetry(0.009)
    assert payload["parts"]["mouth"] == format_symmetry(0.005)


@pytest.mark.parametrize("position", ["左上", "右上", "左下", "右下"])
def test_every_visible_preset_marks_the_hud_visible(position: str) -> None:
    assert _payload(position=position)["visible"] is True


def test_the_hidden_preset_marks_the_hud_invisible() -> None:
    assert _payload(position="非表示")["visible"] is False


def test_an_unknown_preset_hides_rather_than_floating_loose() -> None:
    assert _payload(position="斜め上")["visible"] is False


def test_every_hud_preset_is_accepted_by_the_payload() -> None:
    for position in HUD_POSITION_OPTIONS:
        assert isinstance(build_hud_payload(**{**_payload_kwargs(), "position": position}), dict)


def test_the_hud_preset_list_offers_a_hidden_option_and_defaults_to_top_left() -> None:
    assert HUD_POSITION_OPTIONS[0] == "左上"
    assert HUD_POSITION_OPTIONS[-1] == "非表示"
    assert set(HUD_PLACEMENTS) == set(HUD_POSITION_OPTIONS) - {"非表示"}


# --- §14.3e: 5 Hz の payload を軽いまま保つ -----------------------------------


def test_the_payload_element_is_hidden_and_carries_only_json() -> None:
    """5 Hz で通信を越えるのはこれだけである。"""
    html = hud_html(**_payload_kwargs())

    assert "display:none" in html
    assert html.count("<div") == 1
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["symmetry"] == format_symmetry(0.0215)


def test_the_payload_carries_only_the_current_reading_never_a_whole_series() -> None:
    """折れ線グラフの系列全体ではなく、現在値だけが 5 Hz の payload に乗る。

    「300 点 × 5 系列を毎回送る」実装ではこの assertion が落ちる（罠）。
    数値であるべき箇所（`chart` の各系列値、`symmetry`/`score`/`parts` の
    フォーマット済み文字列）に配列が紛れ込んでいないことを確認する。
    """
    payload = _payload()

    chart = payload["chart"]
    assert isinstance(chart, dict)
    for key in ("overall", "contour", "eyes", "nose", "mouth"):
        assert isinstance(chart[key], (int, float))
    # フォーマット済みの表示値も単一の文字列であり、履歴の配列ではない。
    assert isinstance(payload["symmetry"], str)
    assert isinstance(payload["score"], str)
    for value in payload["parts"].values():
        assert isinstance(value, str)


def test_the_chart_values_reuse_the_same_symmetry_transform_as_the_text() -> None:
    """グラフの数値も `ui.readings.symmetry_percent`（`format_symmetry` と同じ式）
    を経由し、HUD が数値用の 2 つ目の変換へ分岐しないことを確かめる。"""
    from face_symmetry_lab.ui.readings import symmetry_percent

    payload = _payload(
        overall=0.0215, parts={"contour": 0.01, "eyes": 0.02, "nose": None, "mouth": 0.015}
    )

    assert payload["chart"]["overall"] == pytest.approx(symmetry_percent(0.0215))
    assert payload["chart"]["contour"] == pytest.approx(symmetry_percent(0.01))
    assert payload["chart"]["nose"] is None  # 欠測は 0 ではなく欠測のまま運ぶ


def test_the_five_hertz_update_never_ships_css() -> None:
    """5 Hz で stylesheet を送り直すと全文書のスタイル再計算が起き、それが
    映像のカクつきの原因だった。二度と戻してはならない。"""
    html = hud_html(**_payload_kwargs())

    assert "<style" not in html
    assert "backdrop-filter" not in html
    assert "transition" not in html


def test_the_hud_never_uses_backdrop_filter_anywhere() -> None:
    """背景ぼかしは更新ごとではなく *映像フレーム* ごとに再実行される。"""
    assert "backdrop-filter" not in HUD_STYLESHEET
    assert "backdrop-filter" not in hud_bridge_script(fullscreen=False, position="左上")


# --- §14.3e: bridge スクリプト ------------------------------------------------


def test_the_bridge_makes_the_hud_draggable() -> None:
    script = _bridge()

    assert "pointerdown" in script
    assert "pointermove" in script
    assert "setPointerCapture" in script


def test_dragging_clamps_to_the_viewport_on_every_side_not_only_the_top_left() -> None:
    """H1 で HUD の内容量が増えた分だけ、画面外へ完全にドラッグし出せてしまう
    影響も大きくなる。以前は左上だけを ``Math.max(0, ...)`` で制限していたが、
    右端・下端も HUD 自身の幅・高さぶんだけ制限する。
    """
    assert "window.innerWidth - hud.offsetWidth" in HUD_RUNTIME_JS
    assert "window.innerHeight - hud.offsetHeight" in HUD_RUNTIME_JS


def test_the_bridge_attaches_the_hud_to_the_page_body() -> None:
    """<body> 直下なら何にもクリップされず `position: fixed` が期待どおり働く。
    これが以前に全画面で HUD が消えていた理由でもある。"""
    assert "document.body.appendChild(hud)" in HUD_RUNTIME_JS


def test_the_bridge_toggles_fullscreen_by_class_rather_than_reinjecting_css() -> None:
    assert "classList.toggle('fsl-fullscreen', !!CFG.fullscreen)" in _bridge()
    assert "fsl-fullscreen" in _bridge(fullscreen=True)


def test_the_bridge_replaces_its_timer_instead_of_stacking_them() -> None:
    """script は全再実行のたびに走るため、タイマーが積み上がってはならない。"""
    script = _bridge()

    assert "clearInterval" in script
    assert script.count("setInterval") == 1


def test_the_bridge_creates_the_hud_only_once() -> None:
    script = _bridge()

    assert "if (!hud)" in script
    assert "getElementById('fsl-hud')" in script


def test_the_runtime_is_injected_into_the_page_realm_not_the_component_iframe() -> None:
    """コンポーネント iframe 内で定義したタイマーは Streamlit がそれを差し替えた
    時点で死に、タイマー ID だけが親に残って HUD が停止する。"""
    script = _bridge()

    assert "createElement('script')" in script
    assert "fsl-hud-runtime" in script
    assert "doc.head.appendChild(runtime)" in script


# --- H1: 折れ線グラフ ---------------------------------------------------------


def test_the_chart_window_defaults_to_sixty_seconds_at_five_hertz() -> None:
    assert HUD_CHART_WINDOW_SECONDS == 60
    assert HUD_CHART_CAPACITY == HUD_CHART_WINDOW_SECONDS * (1000 // HUD_REFRESH_MS)
    assert HUD_CHART_CAPACITY == 300


def test_the_chart_y_axis_is_a_fixed_window_not_zero_to_a_hundred_or_auto_scaled() -> None:
    """実測値は 96〜99% に集まる（`docs/symmetry-metric-measurements.md` §12.1）。
    0〜100% 固定では 5 本が一直線になり、自動スケールではノイズが誇張される。
    """
    assert HUD_CHART_Y_MIN == 95.0
    assert HUD_CHART_Y_MAX == 100.0


# --- FR-039: 参照平均の併記 ----------------------------------------------------


def _baseline(**means: float) -> BaselineSummary:
    defaults = {
        "score_points": 45.0,
        "symmetry_percent_overall": 96.0,
        "symmetry_percent_contour": 95.0,
        "symmetry_percent_eyes": 96.5,
        "symmetry_percent_nose": 97.0,
        "symmetry_percent_mouth": 96.2,
    }
    defaults.update(means)
    return BaselineSummary(
        age=20,
        gender=GenderLabel.MALE,
        count=75,
        metrics={name: MetricSummary(mean=value, sd=1.0) for name, value in defaults.items()},
    )


def test_the_baseline_is_compared_against_the_same_value_that_is_displayed() -> None:
    """AC-BASELINE-01: 括弧の中の差は、画面に出ている数から作る。"""
    payload = _payload(baseline=_baseline())

    baseline = payload["baseline"]
    assert payload["symmetry"] == "97.85%"
    assert baseline["symmetry"] == "平均 96.00%（+1.85）"


def test_the_baseline_score_is_compared_in_points_and_not_in_percent() -> None:
    """§8.6.7: 点と % を混ぜない（罠: 対称度の目盛りで引き算する実装は 51.9 を出す）。"""
    payload = _payload(baseline=_baseline())

    assert payload["score"] == format_score(0.031)
    assert payload["baseline"]["score"] == "平均 45.0 点（+2.5）"
    assert "%" not in payload["baseline"]["score"]


def test_every_part_carries_its_own_baseline() -> None:
    payload = _payload(baseline=_baseline())

    parts = payload["baseline"]["parts"]
    assert set(parts) == set(PART_LABELS)
    assert parts["contour"] == "平均 95.00%（+4.00）"
    assert parts["nose"] == "平均 97.00%（+0.00）"


def test_the_baseline_note_names_the_population() -> None:
    """AC-BASELINE-04: 何と比べているのかを、併記と同じ吹き出しの中へ置く。"""
    note = _payload(baseline=_baseline())["baseline"]["note"]

    assert "20 歳" in note and "男性" in note and "75 件" in note
    assert "人種で絞り込まず" in note


def test_no_baseline_leaves_every_slot_empty() -> None:
    """併記が無いときは空文字であり `—` ではない（`:empty` で行ごと畳む）。"""
    baseline = _payload()["baseline"]

    assert baseline["symmetry"] == ""
    assert baseline["score"] == ""
    assert baseline["note"] == ""
    assert set(baseline["parts"]) == set(PART_LABELS)
    assert all(text == "" for text in baseline["parts"].values())


def test_the_baseline_never_changes_the_measured_readings() -> None:
    """§10.12-7: 参照統計は表示の上乗せであり、測定値の表示を変えない。"""
    without = _payload()
    with_baseline = _payload(baseline=_baseline())

    for key in ("symmetry", "score", "parts", "chart", "label", "gate"):
        assert without[key] == with_baseline[key]


def test_a_missing_reading_gets_the_mean_without_a_difference() -> None:
    """不変条件 11: `—` の項目に `+0.00` を作らない。"""
    payload = _payload(equal_weight=None, baseline=_baseline())

    assert payload["score"] == "—"
    assert payload["baseline"]["score"] == "平均 45.0 点"


def test_render_hud_payload_resolves_the_baseline_for_the_shown_estimate() -> None:
    """FR-039: 同梱の参照統計を通した、Camera 側の端から端までの配線。"""
    at = AppTest.from_string(
        "import json\n"
        "from datetime import UTC, datetime\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.domain.demographics import build_demographic_estimate\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "estimate = build_demographic_estimate(\n"
        "    estimated_age_years=20.4,\n"
        "    gender_probabilities=(0.2, 0.8),\n"
        "    model_version='age-gender/test',\n"
        "    model_sha256='a' * 64,\n"
        "    observed_at=datetime(2026, 8, 16, tzinfo=UTC),\n"
        ")\n"
        "snapshot = PipelineSnapshot(analysis=outcome, demographic_estimate=estimate)\n"
        "render_hud_payload(snapshot, show_demographics=True)\n"
    )
    at.run()

    assert not at.exception
    payload = json.loads(at.markdown[0].value.split(">", 1)[1].rsplit("<", 1)[0])
    assert "参考年齢 20 歳" in payload["demo"]
    assert "20 歳・男性" in payload["baseline"]["note"]
    assert "75 件" in payload["baseline"]["note"]
    assert payload["baseline"]["symmetry"].startswith("平均 ")


def test_render_hud_payload_drops_the_baseline_when_the_gate_hides_the_age() -> None:
    """§10.12-3: 母集団の年齢からゲートの内部判断を復元させない。"""
    at = AppTest.from_string(
        "import json\n"
        "from datetime import UTC, datetime\n"
        "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.domain.demographics import build_demographic_estimate\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "estimate = build_demographic_estimate(\n"
        "    estimated_age_years=20.4,\n"
        "    gender_probabilities=(0.2, 0.8),\n"
        "    model_version='age-gender/test',\n"
        "    model_sha256='a' * 64,\n"
        "    observed_at=datetime(2026, 8, 16, tzinfo=UTC),\n"
        ")\n"
        "snapshot = PipelineSnapshot(\n"
        "    analysis=outcome,\n"
        "    demographic_estimate=estimate,\n"
        "    age_gate_outcome=AgeGateOutcome(\n"
        "        status=AgeGateStatus.OUT_OF_RANGE, decision=None, reason=None\n"
        "    ),\n"
        ")\n"
        "render_hud_payload(snapshot, show_demographics=True)\n"
    )
    at.run()

    assert not at.exception
    payload = json.loads(at.markdown[0].value.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["baseline"]["note"] == ""
    assert payload["baseline"]["symmetry"] == ""
    assert "20" not in payload["baseline"]["note"]
    # 左右差の表示は変わらない（不変条件 12）。
    assert payload["symmetry"] == format_symmetry(0.021)


# --- FR-040 / §5.4: 状態と平滑化時間のテキスト表示 ----------------------------


def test_the_payload_carries_the_gate_state_as_text() -> None:
    """§5.4: 枠色と同じ情報をテキストでも運ぶ。整形は `ui.readings` と共有する。"""
    payload = _payload(state=gate_label(AnalysisState(status=Status.OK)))

    assert payload["state"] == "計測中"
    assert payload["state"] != payload["gate"], "色と文字は別の経路である"


def test_the_payload_carries_the_active_smoothing_window() -> None:
    """FR-040: いま何秒の平均を見ているかを、数値の隣で読めるようにする。"""
    payload = _payload(smoothing=format_smoothing_window(0.4))

    assert payload["smoothing"] == "平滑化 0.4 秒"


def test_the_state_and_smoothing_default_to_blank_rather_than_a_dash() -> None:
    """併記の不在は測定の欠測ではない（`—` を入れない、`:empty` で畳む）。"""
    payload = _payload()

    assert payload["state"] == ""
    assert payload["smoothing"] == ""


def test_the_state_and_smoothing_never_change_the_measured_readings() -> None:
    """罠: 状態や設定の表示が測定値の整形へ混ざる実装はここで落ちる。"""
    plain = _payload()
    annotated = _payload(
        state=gate_label(AnalysisState(status=Status.HOLD)),
        smoothing=format_smoothing_window(2.0),
    )

    for key in ("symmetry", "score", "parts", "chart"):
        assert annotated[key] == plain[key]


def test_render_hud_payload_publishes_the_state_and_smoothing_window() -> None:
    """FR-040: snapshot の解析状態と、pipeline が報告する時定数を配線する。"""
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "from tests.ui.test_scores import _outcome_for_display, _symmetry_3d_result\n"
        "outcome = _outcome_for_display(\n"
        "    symmetry_3d=_symmetry_3d_result(\n"
        "        contour=0.01, eyes=0.02, nose=0.03, mouth=0.015, overall=0.018\n"
        "    ),\n"
        "    smooth_3d=0.021,\n"
        ")\n"
        "render_hud_payload(\n"
        "    PipelineSnapshot(analysis=outcome),\n"
        "    show_demographics=False,\n"
        "    smoothing_tau_seconds=2.0,\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["smoothing"] == "平滑化 2.0 秒"
    assert payload["state"] in set(GATE_LABELS.values())


def test_render_hud_payload_leaves_the_smoothing_blank_when_it_is_unknown() -> None:
    """analyzer 未接続の pipeline（`smoothing_tau_seconds` が `None`）でも落ちない。"""
    at = AppTest.from_string(
        "import json\n"
        "from face_symmetry_lab.application.realtime import PipelineSnapshot\n"
        "from face_symmetry_lab.ui.hud import render_hud_payload\n"
        "render_hud_payload(PipelineSnapshot(), show_demographics=False)\n"
    )
    at.run()

    assert not at.exception
    html = at.markdown[0].value
    payload = json.loads(html.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["smoothing"] == ""
    assert payload["state"] == "待機中"
