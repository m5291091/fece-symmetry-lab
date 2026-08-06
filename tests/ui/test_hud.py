"""§14.3e の HUD 資産に対するテスト。

ここで検証するのはブラウザへ渡る文字列そのもの（CSS・実行時 JS・5 Hz の
payload）である。HUD をカメラ画面の合成から切り離したので、テストも同じ境界
で分ける。カメラ側の配線（どの fragment が payload を publish するか等）は
`test_camera.py` が受け持つ。
"""

import json

import pytest

from face_symmetry_lab.domain.models import GenderLabel
from face_symmetry_lab.ui.hud import (
    FULLSCREEN_BADGE,
    FULLSCREEN_HINT,
    HUD_PLACEMENTS,
    HUD_POSITION_OPTIONS,
    HUD_RUNTIME_JS,
    HUD_STYLESHEET,
    VIDEO_CHROME_CSS,
    build_hud_payload,
    hud_bridge_script,
    hud_html,
)
from face_symmetry_lab.ui.scores import GENDER_LABELS, format_symmetry


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
        "estimate": None,
        "show_demographics": False,
        "gate": "#2E9E5B",
        "position": "左上",
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


def test_a_missing_symmetry_value_renders_as_a_dash() -> None:
    """NFR-013: 欠測は欠測として示し、古い値や 0 を出さない。"""
    assert _payload(overall=None)["symmetry"] == "—"
    assert "—" in hud_html(**{**_payload_kwargs(), "overall": None})


def test_the_hud_carries_the_experimental_label() -> None:
    assert _payload()["label"] == "対称度（実験値）"


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
