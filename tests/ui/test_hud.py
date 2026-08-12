"""§14.3e の HUD 資産に対するテスト。

ここで検証するのはブラウザへ渡る文字列そのもの（CSS・実行時 JS・5 Hz の
payload）である。HUD をカメラ画面の合成から切り離したので、テストも同じ境界
で分ける。カメラ側の配線（どの fragment が payload を publish するか等）は
`test_camera.py` が受け持つ。
"""

import json
import re

import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.models import GenderLabel
from face_symmetry_lab.ui.hud import (
    FULLSCREEN_BADGE,
    FULLSCREEN_HINT,
    HUD_CHART_CAPACITY,
    HUD_CHART_WINDOW_SECONDS,
    HUD_CHART_Y_MAX,
    HUD_CHART_Y_MIN,
    HUD_PLACEMENTS,
    HUD_POSITION_OPTIONS,
    HUD_REFRESH_MS,
    HUD_RUNTIME_JS,
    HUD_STYLESHEET,
    VIDEO_CHROME_CSS,
    build_hud_payload,
    hud_bridge_script,
    hud_html,
)
from face_symmetry_lab.ui.scores import (
    GENDER_LABELS,
    PART_LABELS,
    format_score,
    format_symmetry,
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
    """グラフの数値も `ui.scores.symmetry_percent`（`format_symmetry` と同じ式）
    を経由し、HUD が数値用の 2 つ目の変換へ分岐しないことを確かめる。"""
    from face_symmetry_lab.ui.scores import symmetry_percent

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


# --- §14.3e: スタイルシート ---------------------------------------------------


def test_the_hud_shell_stays_click_through_except_for_its_grip() -> None:
    """HUD は映像自身の STOP 操作子の上に重なる。"""
    assert "#fsl-hud {" in HUD_STYLESHEET
    assert "pointer-events: none;" in HUD_STYLESHEET
    # grip だけが操作を受け取るので、HUD をドラッグできる。
    assert "#fsl-hud-grip" in HUD_STYLESHEET
    assert "pointer-events: auto;" in HUD_STYLESHEET


def test_the_hud_keeps_digit_widths_stable_at_five_hertz() -> None:
    assert "tabular-nums" in HUD_STYLESHEET


def test_fullscreen_targets_the_video_through_a_stable_hook() -> None:
    """以前の `data-testid` セレクタは何にも一致せず、HUD だけが動いていた。"""
    assert ".st-key-camera_preview_frame iframe" in HUD_STYLESHEET
    assert "data-testid" not in HUD_STYLESHEET


def test_no_streamlit_internal_testid_is_relied_on_any_more() -> None:
    """`data-testid` の利用は全画面表示を黙って壊した。もう何も選択に使わない。

    ブラウザが実際に評価する文字列である CSS と bridge script に対して確認する。
    そうすることで、それを外した理由を説明するコメントがこのテストへ引っかからない。
    """
    assert "data-testid" not in HUD_STYLESHEET
    assert "data-testid" not in hud_bridge_script(fullscreen=True, position="左上")


def test_the_overlay_css_avoids_structural_selectors() -> None:
    """§5.2 の例外は安定した手がかりを要求する。`nth-child` の位置指定は不可。"""
    assert "nth-child" not in HUD_STYLESHEET
    assert "nth-of-type" not in HUD_STYLESHEET


def test_fullscreen_stacks_above_streamlit_chrome() -> None:
    """Streamlit の sidebar/header は z-index 999990 以上にあるため、それより
    低いものは覆われる。実際に映像の 1/3 と HUD 全体が隠れていた。"""
    assert "body.fsl-fullscreen .stSidebar" in HUD_STYLESHEET
    assert "body.fsl-fullscreen .stAppHeader" in HUD_STYLESHEET
    assert "display: none !important;" in HUD_STYLESHEET
    # 映像 < HUD < 終了ボタンの順で、すべて Streamlit 自身の装飾より上に置く。
    for value in (1000000, 1000050, 1000060):
        assert f"z-index: {value}" in HUD_STYLESHEET


def _z_index_of(selector: str) -> int:
    """CSS 規則 ``selector { ... z-index: N ...}`` の ``N`` を取り出す。"""
    start = HUD_STYLESHEET.index(selector)
    body = HUD_STYLESHEET[start : HUD_STYLESHEET.index("}", start)]
    match = re.search(r"z-index:\s*(\d+)", body)
    assert match is not None, selector
    return int(match.group(1))


def test_pseudo_fullscreen_already_keeps_the_hud_above_the_video_iframe() -> None:
    """設計3（H1）: HUD は `document.body` 直下に z-index 1000050 で固定され、
    擬似全画面の映像 iframe は 1000000 である。数値上 HUD は常に映像より上に
    あり、H1 で内容を拡張する前から擬似全画面でも表示されていた——新規の実装
    ではなく、この重ね順を固定する回帰テストである。
    """
    hud_z = _z_index_of("#fsl-hud {")
    video_z = _z_index_of("body.fsl-fullscreen .st-key-camera_preview_frame iframe {")

    assert hud_z > video_z


def test_the_hud_shell_caps_its_size_for_small_or_fullscreen_viewports() -> None:
    """H1 で内容量が増えるため、画面外へあふれないよう上限を設ける。"""
    hud_rule = HUD_STYLESHEET[
        HUD_STYLESHEET.index("#fsl-hud {") : HUD_STYLESHEET.index("#fsl-hud[hidden]")
    ]

    assert "max-width" in hud_rule
    assert "max-height" in hud_rule
    assert "overflow-y: auto" in hud_rule


def test_the_exit_control_shrinks_to_its_button() -> None:
    """Streamlit の縦ブロックは width:100% なので、`right` で固定したコンテナは
    ビューポート全体へ広がり、ボタンを左端の外（実測 x=-24）へ落としていた。"""
    assert "width: auto !important;" in HUD_STYLESHEET
    assert "left: auto;" in HUD_STYLESHEET


def test_the_band_never_intercepts_clicks() -> None:
    band = HUD_STYLESHEET[
        HUD_STYLESHEET.index("#fsl-chrome {") : HUD_STYLESHEET.index("#fsl-chrome[hidden]")
    ]

    assert "pointer-events: none;" in band


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


# --- §14.3e: 実行時コード -----------------------------------------------------


def test_the_runtime_owns_the_timer_and_the_element() -> None:
    assert "setInterval(sync" in HUD_RUNTIME_JS
    assert "document.body.appendChild(hud)" in HUD_RUNTIME_JS
    assert "clearInterval(window.__fslHudTimer)" in HUD_RUNTIME_JS


def test_the_hud_disappears_when_the_camera_is_not_running() -> None:
    """payload を publish するパネルは再生中しか描画されない。payload が無い
    ときは直前の値を凍結表示せず HUD を隠さなければならない。"""
    branch = HUD_RUNTIME_JS[
        HUD_RUNTIME_JS.index("if (!holder)") : HUD_RUNTIME_JS.index("if (hint) hint.hidden = true;")
    ]

    assert "hudEl.hidden = true;" in branch
    assert "return;" in branch


def test_the_component_iframe_access_is_guarded() -> None:
    """cross-origin や未準備の iframe が毎 tick 例外を投げてはならない。"""
    assert "try { vdoc = host && host.contentDocument; } catch (err) { return; }" in HUD_RUNTIME_JS


def test_the_control_row_is_tagged_from_js_rather_than_by_a_vendor_class() -> None:
    """コンポーネントは emotion/MUI のハッシュしか出さないので、アプリが自前で付ける。"""
    assert "classList.add('fsl-controls')" in HUD_RUNTIME_JS


def test_the_control_row_never_wraps_the_video() -> None:
    """タグ付けは映像を含まない最後のラッパーで止め、再生開始で木構造が変わって
    映像を含むようになったラッパーからは class を外す。"""
    assert "if (n.querySelector('video')) break;" in HUD_RUNTIME_JS
    assert "if (stale[s].querySelector('video')) stale[s].classList.remove('fsl-controls');" in (
        HUD_RUNTIME_JS
    )


def test_the_band_is_built_and_driven_by_the_runtime() -> None:
    assert "fsl-chrome" in HUD_RUNTIME_JS
    assert "chrome.hidden = !CFG.fullscreen;" in HUD_RUNTIME_JS
    # hint は開始のための案内なので、カメラが動き出したら消える。
    assert "hint.hidden = true;" in HUD_RUNTIME_JS
    assert "hint.hidden = false;" in HUD_RUNTIME_JS


def test_the_fullscreen_band_states_where_you_are_and_how_to_leave() -> None:
    """全画面はページの装飾を畳むため、これが唯一残る文章である。"""
    assert "全画面表示中" in FULLSCREEN_BADGE
    assert "終了" in FULLSCREEN_BADGE
    assert "START" in FULLSCREEN_HINT
    assert "F11" in FULLSCREEN_HINT


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


def test_the_chart_skeleton_is_built_once_with_five_polylines_axis_labels_and_a_legend() -> None:
    """<svg>・<polyline> 5 本・軸ラベル・凡例は bridge が 1 度だけ作る骨格に含む。"""
    skeleton = HUD_RUNTIME_JS[
        HUD_RUNTIME_JS.index("if (!hud)") : HUD_RUNTIME_JS.index("document.body.appendChild(hud)")
    ]

    assert skeleton.count("<svg") == 1
    assert skeleton.count("<polyline") == 5
    for name in ("overall", "contour", "eyes", "nose", "mouth"):
        assert f"fsl-chart-{name}" in skeleton
    # 軸ラベル（95%・100%）と凡例。縦軸の幅を利用者が誤解しないための明示。
    assert "CFG.chartYMin" in skeleton or "chartYMin" in skeleton
    assert "CFG.chartYMax" in skeleton or "chartYMax" in skeleton
    assert "fsl-hud-legend" in skeleton


def test_the_five_hertz_tick_only_rewrites_the_points_attribute() -> None:
    """5 Hz の `sync` 本体は、骨格を作り直さず `points` 属性だけを書き換える。"""
    sync_body = HUD_RUNTIME_JS[
        HUD_RUNTIME_JS.index("var sync = function () {") : HUD_RUNTIME_JS.index(
            "sync();\n  window.__fslHudTimer"
        )
    ]

    assert "createElement('svg')" not in sync_body
    assert "innerHTML" not in sync_body
    assert "setAttribute('points'" in sync_body


def test_the_series_ring_buffer_lives_in_memory_not_in_local_storage() -> None:
    """§3.4: 折れ線グラフの系列は左右差の派生値そのものであり、`fslHudPos`
    （顔由来ではない HUD 位置）とは異なり永続化の禁止対象内である。"""
    assert "window.__fslHudSeries" in HUD_RUNTIME_JS

    # localStorage への書き込み・読み出しは既存の `fslHudPos`（ドラッグ位置）
    # だけであることを、実際に使われているキーを列挙して確かめる（罠 (b):
    # 系列を `localStorage` へも書く実装ではここが `fslHudPos` 以外のキーを
    # 拾って落ちる）。
    keys = re.findall(r"localStorage\.(?:setItem|getItem|removeItem)\(\s*'([^']+)'", HUD_RUNTIME_JS)
    assert keys, "localStorage の呼び出しが見つかりません"
    assert set(keys) == {"fslHudPos"}


def test_the_generation_change_discards_the_ring_buffer_before_the_next_point() -> None:
    """世代識別子（`data.gen`）が変わったら、新しい値を積む前に系列を空にする
    （§8.8 の EMA リセット規約が画面の上で破られないようにするため）。"""
    sync_body = HUD_RUNTIME_JS[
        HUD_RUNTIME_JS.index("var sync = function () {") : HUD_RUNTIME_JS.index(
            "sync();\n  window.__fslHudTimer"
        )
    ]

    assert "series.gen" in sync_body
    assert "data.gen" in sync_body
    # 世代が変わったときに空の buffers を作り直している (0 系列からの再開)。
    assert re.search(r"buffers\s*=\s*\{", sync_body) is not None


def test_missing_chart_values_are_skipped_rather_than_drawn_as_zero() -> None:
    """不変条件 11: 欠測 tick はその系列へ追加しない（0 で埋めない）。"""
    sync_body = HUD_RUNTIME_JS[
        HUD_RUNTIME_JS.index("var sync = function () {") : HUD_RUNTIME_JS.index(
            "sync();\n  window.__fslHudTimer"
        )
    ]

    assert "isFinite" in sync_body or "typeof" in sync_body


# --- 映像フレーム側の CSS -----------------------------------------------------


def test_fullscreen_makes_the_picture_fill_the_frame() -> None:
    """コンポーネントは video を `width: 100%` かつ高さ auto で配置するため、
    全高フレームでは上端の帯状のままだった。"""
    assert "html.fsl-fs video" in VIDEO_CHROME_CSS
    assert "height: 100vh !important;" in VIDEO_CHROME_CSS
    assert "object-fit: contain;" in VIDEO_CHROME_CSS


def test_fullscreen_centres_and_enlarges_the_start_stop_row() -> None:
    """全画面ビューポートの左端では、この行は実際に押しにくかった。"""
    assert "html.fsl-fs .fsl-controls" in VIDEO_CHROME_CSS
    assert "width: max-content;" in VIDEO_CHROME_CSS  # transform ではなく auto margin で中央寄せ
    assert "bottom: 44px;" in VIDEO_CHROME_CSS
    assert "min-height: 48px" in VIDEO_CHROME_CSS


def test_the_control_row_is_centred_without_a_transform() -> None:
    """transform を持つ要素は *fixed* な子孫の包含ブロックになる。

    この行が <video> を包んでいた場合、映像はビューポートではなく pill を基準に
    配置され、隅へ寄ってしまっていた。
    """
    rule = VIDEO_CHROME_CSS[VIDEO_CHROME_CSS.index("html.fsl-fs .fsl-controls {") :]
    rule = rule[: rule.index("}")]

    assert "translateX" not in rule
    assert "transform: none;" in rule
    assert "margin-left: auto;" in rule and "margin-right: auto;" in rule
