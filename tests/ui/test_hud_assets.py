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


def _rule_body(selector: str) -> str:
    """CSS 規則 ``selector { ... }`` の中身だけを取り出す。"""
    start = HUD_STYLESHEET.index(selector)
    return HUD_STYLESHEET[start : HUD_STYLESHEET.index("}", start)]


def test_the_hud_box_scales_with_the_viewport_instead_of_a_fixed_width() -> None:
    """§5.1: 画面が小さいほど HUD も小さくなる。

    **HUD は `pointer-events: none` である。** 内容が箱からあふれても利用者は
    スクロールできず、あふれた数値は読む手段が無いまま消える。したがって
    「あふれたらスクロールさせる」ではなく「どの画面幅でも収まる」で解く。
    罠: `max-width: min(360px, 92vw)` だけに戻すと、狭い画面では箱が
    ビューポートのほぼ全幅を占めたまま中身の文字だけが固定サイズで残る。
    """
    hud_rule = _rule_body("#fsl-hud {")

    assert "clamp(" in hud_rule, "幅を viewport へ追従させること"
    assert "vw" in hud_rule


def test_every_hud_reading_uses_a_fluid_type_size() -> None:
    """画面に出る数値はすべて画面幅へ追従する（§5.1・§5.4）。

    罠: `font-size: 3.2rem` のような固定値へ戻した実装は、狭い画面で
    「98.42%」の末尾が箱の外へ出る。
    """
    for selector in (
        "#fsl-hud-value {",
        "#fsl-hud-label {",
        "#fsl-hud-score-row {",
        "#fsl-hud-parts {",
        "#fsl-hud-demo {",
    ):
        body = _rule_body(selector)
        assert "font-size: clamp(" in body, selector


def test_the_hud_type_also_answers_to_a_short_window() -> None:
    """§5.1: 足りないほうの寸法に合わせる。

    HUD の高さ上限は `90vh` である。横に広く縦に短い窓（ノート PC の分割表示
    など）では、幅だけを見て決めた文字サイズだと中身が縦にあふれ、
    `pointer-events: none` の箱では読む手段が無くなる。幅と高さの小さいほうへ
    合わせる。罠: `vh` の項を落とすと、この assertion が落ちる。
    """
    body = _rule_body("#fsl-hud-value {")

    assert "vh" in body
    assert "min(" in body


def test_long_readings_wrap_rather_than_disappear() -> None:
    """あふれるくらいなら折り返す。`pointer-events: none` の箱では、はみ出した
    分を利用者が引き寄せる手段が無いためである。"""
    assert "overflow-wrap" in _rule_body("#fsl-hud {")


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


# --- FR-039: 参照平均の併記 ----------------------------------------------------


def test_the_baseline_rows_scale_with_the_hud() -> None:
    """§5.1: 併記も箱と一緒に縮む（固定 px へ戻すと狭い画面で外へ出る）。"""
    rule = HUD_STYLESHEET[HUD_STYLESHEET.index(".fsl-hud-baseline {") :]
    rule = rule[: rule.index("}")]

    assert "font-size: clamp(" in rule
    assert "min(" in rule  # 幅と高さの足りないほうへ合わせる


def test_an_absent_baseline_collapses_instead_of_leaving_a_blank_row() -> None:
    """FR-024 と同じ理由: 空行は「動いていて失敗している」と読まれる。"""
    assert ".fsl-hud-baseline:empty { display: none; }" in HUD_STYLESHEET
    assert "#fsl-hud-baseline-note:empty { display: none; }" in HUD_STYLESHEET


def test_the_baseline_is_quieter_than_the_reading_it_annotates() -> None:
    """§5.2: 差を達成度として演出しない。比べる相手であって主役ではない。"""
    rule = HUD_STYLESHEET[HUD_STYLESHEET.index(".fsl-hud-baseline {") :]
    rule = rule[: rule.index("}")]

    assert "opacity: .66;" in rule
    # 色・矢印・ゲージで大小を演出しない。
    for banned in ("color:", "background", "::before", "transform"):
        assert banned not in rule


def test_each_part_keeps_its_own_baseline_beneath_it() -> None:
    """4 部位 × 2 値が 1 行に混ざると、どの平均がどの部位のものか読めない。"""
    assert ".fsl-hud-part { display: flex; flex-direction: column; }" in HUD_STYLESHEET
    for part in ("contour", "eyes", "nose", "mouth"):
        assert f'id="fsl-hud-baseline-{part}"' in HUD_RUNTIME_JS


def test_the_baseline_skeleton_is_built_once_with_the_rest_of_the_hud() -> None:
    """5 Hz の tick は骨格を作り直さない（§5.2）。"""
    for element_id in (
        "fsl-hud-baseline-symmetry",
        "fsl-hud-baseline-score",
        "fsl-hud-baseline-note",
    ):
        assert f'id="{element_id}"' in HUD_RUNTIME_JS
        assert f"setText('{element_id}'" in HUD_RUNTIME_JS


def test_the_tick_reads_the_baseline_from_the_payload() -> None:
    """罠: HUD 側で平均を組み立て直す実装は、吹き出しと 2 つ目の表示規則へ分岐する。"""
    assert "var baseline = data.baseline || {};" in HUD_RUNTIME_JS
    assert "baseline.symmetry" in HUD_RUNTIME_JS
    assert "baseline.score" in HUD_RUNTIME_JS
    assert "baseline.note" in HUD_RUNTIME_JS
    # 併記の不在に `—` を入れない（測定の欠測ではない）。
    assert "baseline.symmetry || ''" in HUD_RUNTIME_JS


def test_the_note_never_pushes_the_drag_grip_out_of_the_hud() -> None:
    """HUD も `pointer-events: none` + `max-height` である（吹き出しと同じ理由）。"""
    assert HUD_RUNTIME_JS.index("fsl-hud-grip") < HUD_RUNTIME_JS.index('id="fsl-hud-baseline-note"')


def test_the_hud_footprint_includes_its_padding_and_border() -> None:
    """`width: clamp(210px, 26vw, 360px)` を**見た目の実寸**にする。

    吹き出し（`ui.still_overlay`）と同じ理由である。既定の `content-box` では
    padding（左右 13〜22px）と border 2px×2 が宣言幅の外側へ足され、実測
    （headless Chrome、ビューポート 800px）で 210px 指定に対して 240px あった。
    箱の幅は映像を覆う面積そのものなので、宣言値と実寸をそろえる。
    """
    rule = HUD_STYLESHEET[
        HUD_STYLESHEET.index("#fsl-hud {") : HUD_STYLESHEET.index("#fsl-hud[hidden]")
    ]

    assert "box-sizing: border-box;" in rule
    assert "width: clamp(210px, 26vw, 360px);" in rule


# --- FR-040 / §5.4: 状態と平滑化時間のチップ ----------------------------------


def test_the_state_and_smoothing_chips_are_part_of_the_skeleton() -> None:
    """骨格は bridge が 1 度だけ作る。5Hz の tick で作り直さない（§5.2）。"""
    skeleton = HUD_RUNTIME_JS.split("document.body.appendChild(hud);")[0]

    assert 'id="fsl-hud-chips"' in skeleton
    assert 'id="fsl-hud-state"' in skeleton
    assert 'id="fsl-hud-smoothing"' in skeleton


def test_the_tick_only_writes_the_chip_text() -> None:
    """罠: tick で innerHTML を組み直す実装は、毎秒 5 回の style 再計算を招く。"""
    tick = HUD_RUNTIME_JS.split("var sync = function ()")[1]

    assert "setText('fsl-hud-state', data.state" in tick
    assert "setText('fsl-hud-smoothing', data.smoothing" in tick
    assert "fsl-hud-chips'" not in tick, "チップの器そのものは tick で触らない"


def test_an_empty_chip_collapses_instead_of_leaving_a_blank_pill() -> None:
    """設定や状態が無いときに、空の丸だけが映像の上へ残らないようにする。"""
    assert ".fsl-hud-chip:empty { display: none; }" in HUD_STYLESHEET
    assert "#fsl-hud-chips:empty" in HUD_STYLESHEET


def test_the_chips_scale_with_the_hud_like_every_other_reading() -> None:
    """狭い画面で箱だけが縮み、文字が元の大きさで残ることがないようにする。"""
    block = HUD_STYLESHEET.split(".fsl-hud-chip {")[1].split("}")[0]

    assert "font-size: clamp(" in block


def test_the_chips_never_repaint_themselves_with_the_gate_colour() -> None:
    """§5.4: 状態を色で二重に語らない。枠色は箱の border が既に担っている。

    罠: チップの背景を `data.gate` で塗る実装は、色を落とすと情報が消える
    （テキストだけを手掛かりにしている利用者に何も残らない）。
    """
    tick = HUD_RUNTIME_JS.split("var sync = function ()")[1]

    assert "fsl-hud-state').style" not in tick
    assert "stateEl.style" not in tick


# --- H5: 折れ線グラフの面 -----------------------------------------------------


def test_the_overall_series_carries_a_filled_area_built_once() -> None:
    """全体の線だけを面で塗り、5 本のうちどれが総合値かを一目で分かるようにする。"""
    skeleton = HUD_RUNTIME_JS.split("document.body.appendChild(hud);")[0]

    assert 'id="fsl-chart-overall-area"' in skeleton
    assert "<polygon" in skeleton
    assert ".fsl-chart-area" in HUD_STYLESHEET


def test_the_area_is_redrawn_from_the_same_buffer_as_the_line() -> None:
    """罠: 面だけ別の系列・別の変換で描く実装は、線と面がずれる。"""
    tick = HUD_RUNTIME_JS.split("var sync = function ()")[1]

    assert "fsl-chart-overall-area" in tick
    assert "areaPointsFor(" in tick


def test_the_area_stays_behind_the_lines() -> None:
    """SVG は文書順で重なる。面が線より後に来ると、線を覆い隠す。"""
    skeleton = HUD_RUNTIME_JS.split("document.body.appendChild(hud);")[0]

    assert skeleton.index('id="fsl-chart-overall-area"') < skeleton.index('id="fsl-chart-overall"')


def test_the_display_hold_has_its_own_chip_in_the_skeleton() -> None:
    """FR-041: 数値が止まっているのが設定によるものだと画面から読めるようにする。"""
    skeleton = HUD_RUNTIME_JS.split("document.body.appendChild(hud);")[0]
    tick = HUD_RUNTIME_JS.split("var sync = function ()")[1]

    assert 'id="fsl-hud-hold"' in skeleton
    assert "setText('fsl-hud-hold', data.hold" in tick


# --- FR-043: UI 表示言語の切替 -------------------------------------------------


def test_the_static_labels_are_read_from_the_bridge_config_not_baked_into_the_skeleton() -> None:
    """FR-043: 骨格（`HUD_RUNTIME_JS`）は 1 度しか作り直さないので、日本語を
    直接焼き込むと言語切替が効かない。CFG から読む配線があることを固定する。
    """
    assert "CFG.scoreLabel" in HUD_RUNTIME_JS
    assert "CFG.partLabels.contour" in HUD_RUNTIME_JS
    assert "CFG.partLabels.eyes" in HUD_RUNTIME_JS
    assert "CFG.partLabels.nose" in HUD_RUNTIME_JS
    assert "CFG.partLabels.mouth" in HUD_RUNTIME_JS
    assert "CFG.legendOverallLabel" in HUD_RUNTIME_JS
    assert "CFG.gripText" in HUD_RUNTIME_JS
    # このラベル書き換えは、骨格を新規に作った・作らなかったに関わらず毎回走る
    # 節に置く（`if (!hud) { ... }` の外）。bridge の再実行だけで言語が切り替わる
    # ようにするためである。
    assert HUD_RUNTIME_JS.index("CFG.scoreLabel") > HUD_RUNTIME_JS.index(
        "document.body.appendChild(hud);"
    )


def test_the_runtime_no_longer_hardcodes_the_japanese_labels() -> None:
    """罠: 骨格へ日本語を戻すと、英語モードでも「輪郭」等が焼き付いたまま出る。"""
    assert '<span id="fsl-hud-score-label">全体のスコア</span>' not in HUD_RUNTIME_JS
    assert "<b>輪郭</b>" not in HUD_RUNTIME_JS
    assert ">全体</span>" not in HUD_RUNTIME_JS
    assert '<div id="fsl-hud-grip">⠿ ドラッグで移動</div>' not in HUD_RUNTIME_JS
