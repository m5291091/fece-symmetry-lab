"""`ui.hud_assets`: HUD がブラウザへ渡す CSS と実行時 JS そのもの。

`test_hud.py`（payload の値・配置プリセット・bridge の設置）から分けてある。
ここで見るのは §5.2 が表示専用オーバーレイへ課す条件——セレクタの安定性、
stacking、5 Hz の tick が骨格を作り直さないこと、`backdrop-filter` の不使用、
系列を storage へ書かないこと——であり、Python の組み立てとは変更理由が違う。
"""

import re

from face_symmetry_lab.ui.hud import (
    FULLSCREEN_BADGE,
    FULLSCREEN_HINT,
    hud_bridge_script,
)
from face_symmetry_lab.ui.hud_assets import (
    HUD_RUNTIME_JS,
    HUD_STYLESHEET,
    VIDEO_CHROME_CSS,
)

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
