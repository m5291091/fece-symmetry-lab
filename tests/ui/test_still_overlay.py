"""§5.1/§14.3e: 静止画の吹き出し（2026-08-12、AC-LAYOUT-01）。

罠の置き方: 表示文字列を `build_hud_payload` から取らず独自に整形した実装は
`test_the_still_overlay_reuses_the_hud_display_strings` で落ちる。折れ線グラフを
持ち込んだ実装は `test_the_still_overlay_has_no_chart` で落ちる。
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from face_symmetry_lab.domain.demographics import build_demographic_estimate
from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.ui.hud import build_hud_payload
from face_symmetry_lab.ui.scores import PART_LABELS, format_score, format_symmetry
from face_symmetry_lab.ui.still_overlay import (
    STILL_OVERLAY_CONTAINER_KEY,
    still_overlay_html,
)


def _estimate(age: float = 34.0, female: float = 0.8):
    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(female, 1.0 - female),
        model_version="age-gender/test",
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def _overlay_kwargs() -> dict:
    return {
        "overall": 0.0215,
        "equal_weight": 0.031,
        "parts": {"contour": 0.01, "eyes": 0.02, "nose": 0.03, "mouth": 0.015},
        "estimate": None,
        "show_demographics": False,
        "gate": "#2E9E5B",
    }


def _overlay(**overrides) -> str:
    settings = _overlay_kwargs()
    settings.update(overrides)
    return still_overlay_html(**settings)


# --- 表示文字列は HUD と同じ経路から来る -------------------------------------


def test_the_still_overlay_reuses_the_hud_display_strings() -> None:
    """§14.3e: 同じ測定値に対する 2 つ目の表示規則を作らない。

    罠: `f"{value:.2f}%"` のような独自整形を書いた実装はここで落ちる。
    """
    settings = _overlay_kwargs()
    payload = build_hud_payload(**settings, position="左上", generation="")
    markup = _overlay()

    assert payload["symmetry"] in markup
    assert payload["score"] in markup
    for reading in payload["parts"].values():  # type: ignore[union-attr]
        assert reading in markup


def test_the_still_overlay_shows_the_score_on_the_point_scale() -> None:
    """§8.6.7: 「全体のスコア」は点であり、対称度 % ではない。"""
    markup = _overlay()

    assert format_score(0.031) in markup
    assert format_symmetry(0.031) not in markup


def test_the_still_overlay_labels_all_four_parts() -> None:
    markup = _overlay()

    for name in PART_NAMES:
        assert PART_LABELS[name] in markup


# --- 静止画に固有の制約 -------------------------------------------------------


def test_the_still_overlay_has_no_chart() -> None:
    """§5.1/§14.3e: 静止画は時系列を持たないため、描く系列が存在しない。

    罠: HUD の markup をそのまま流用してグラフごと持ち込む実装は落ちる。
    """
    markup = _overlay()

    for chart_token in ("<svg", "polyline", "viewBox", "凡例", "fsl-legend"):
        assert chart_token not in markup


def test_the_still_overlay_markup_carries_no_script_or_timer() -> None:
    """§14.3e: 吹き出しの markup 自体はコードを含まない。

    ドラッグの実行時コードは bridge（`still_drag_bridge_script`）だけが持ち、
    5Hz の payload に相当するタイマーも配置プリセットも持たない。
    """
    markup = _overlay()

    for token in ("<script", "localStorage", "setInterval", "非表示", "左上"):
        assert token not in markup


# --- 欠測と参考値の扱い -------------------------------------------------------


def test_missing_values_render_as_a_dash_rather_than_zero() -> None:
    """不変条件 11/NFR-013: 欠測は `—`。0 や前回値で埋めない。"""
    markup = _overlay(overall=None, equal_weight=None, parts=dict.fromkeys(PART_NAMES))

    assert "—" in markup
    assert "0.00%" not in markup
    assert "0.0 点" not in markup


def test_the_demographic_row_is_absent_rather_than_empty_when_not_requested() -> None:
    """FR-024: 空行は「動いていて失敗している」と受け取られる。行ごと出さない。"""
    markup = _overlay(estimate=_estimate(), show_demographics=False)

    assert "参考年齢" not in markup
    assert "参考性別" not in markup


def test_the_demographic_row_sits_in_its_own_element_apart_from_the_symmetry() -> None:
    """不変条件 5: 参考年齢・参考性別と左右差を同じ行へ混ぜない。

    左右差の各要素と参考値の要素が別々のタグに入っていることを、要素の切れ目で
    確認する。同じ 1 つのタグへ両方を書き込んだ実装はここで落ちる。
    """
    markup = _overlay(estimate=_estimate(), show_demographics=True)

    demographic_block = markup.split('class="fsl-still-demo"', 1)[1]
    assert "参考年齢" in demographic_block
    assert "参考性別" in demographic_block
    assert format_symmetry(0.0215) not in demographic_block
    assert format_score(0.031) not in demographic_block


def test_a_notice_replaces_the_demographic_row_when_demographics_are_hidden() -> None:
    """FR-038: 参考年齢を表示できないとき、`notice` を吹き出しの独立行へ出す。"""
    markup = _overlay(estimate=None, show_demographics=False, notice="この画像では表示できません")

    assert "この画像では表示できません" in markup


def test_a_notice_is_ignored_while_demographics_are_shown() -> None:
    markup = _overlay(estimate=_estimate(), show_demographics=True, notice="出てはいけない通知")

    assert "出てはいけない通知" not in markup
    assert "参考年齢" in markup


# --- 枠色と文言 ---------------------------------------------------------------


def test_the_gate_colour_reaches_the_overlay_border() -> None:
    """§14.3e: 枠色は `ui.scores.gate_color` が返す状態の識別子である。"""
    assert "#B4433B" in _overlay(gate="#B4433B")


def test_the_overlay_wording_avoids_value_judgements() -> None:
    """§5.3/不変条件 8: 吹き出しにも同じ禁止語が掛かる。"""
    markup = _overlay(estimate=_estimate(), show_demographics=True)

    for forbidden in ("良い", "悪い", "美しい", "優秀", "優れ", "満点", "合格", "理想"):
        assert forbidden not in markup


def test_the_overlay_keeps_the_experimental_label() -> None:
    """不変条件 8: 対称度は「試験中の指標」であることを画面に残す。"""
    assert "実験値" in _overlay()


# --- 描画（キー付きコンテナの中へ重ねる） ------------------------------------


def _rendered_html(at: AppTest) -> str:
    """吹き出しの markup を集める。

    `st.html` ではなく `st.markdown(unsafe_allow_html=True)` を通すので、
    `at.markdown` から読む（理由は `still_overlay` の docstring と、下の
    `test_the_overlay_stylesheet_survives_the_render` を参照）。
    """
    return " ".join(element.value for element in at.markdown)


def _render_source(*, show_demographics: bool = False) -> str:
    return (
        "import numpy as np\n"
        "from face_symmetry_lab.ui.still_overlay import render_image_with_overlay\n"
        "render_image_with_overlay(\n"
        "    np.zeros((8, 8, 3), dtype=np.uint8),\n"
        "    caption='検出したランドマーク',\n"
        "    overall=0.0215,\n"
        "    equal_weight=0.031,\n"
        "    parts={'contour': 0.01, 'eyes': 0.02, 'nose': 0.03, 'mouth': 0.015},\n"
        "    estimate=None,\n"
        f"    show_demographics={show_demographics},\n"
        "    gate='#2E9E5B',\n"
        ")\n"
    )


def test_the_overlay_is_rendered_inside_the_keyed_positioning_context() -> None:
    """§14.3e: 位置指定コンテキストは `st.container(key=...)` 由来の class で作る。

    構造依存セレクタ（`div > div:nth-child(2)` のような）を使わない。Streamlit の
    内部構造が変わると黙って壊れるためである（HUD が `.st-key-camera_preview_frame`
    を使うのと同じ理由）。
    """
    at = AppTest.from_string(_render_source())
    at.run()

    assert not at.exception
    markup = _rendered_html(at)
    assert f"st-key-{STILL_OVERLAY_CONTAINER_KEY}" in markup
    assert "nth-child" not in markup


def test_the_image_and_the_overlay_are_both_drawn() -> None:
    at = AppTest.from_string(_render_source())
    at.run()

    assert not at.exception
    assert len(at.get("image")) == 1
    assert format_score(0.031) in _rendered_html(at)


def test_the_preview_must_be_an_image_array() -> None:
    """`ui` は幾何計算を持たないが、渡されたものが画像であることは確かめる。"""
    with pytest.raises(ValueError, match="画像"):
        from face_symmetry_lab.ui.still_overlay import render_image_with_overlay

        render_image_with_overlay(
            np.zeros((8, 8), dtype=np.uint8),
            caption="c",
            overall=None,
            equal_weight=None,
            parts={},
            estimate=None,
            show_demographics=False,
            gate="#2E9E5B",
        )


# --- 見出し結果（2026-08-12 に削除）から引き継いだ不変条件 --------------------


def test_the_score_and_the_symmetry_are_distinct_readings() -> None:
    """§8.6.5: 「全体のスコア」（`s_3d`）と「全体の対称度」（`d_3d`）は重みだけが
    違う別の総合量で、一致しない。罠: 吹き出しへ `d_3d` を両方へ渡す実装は、
    2 つの読みが同じ量の 2 つの表示になっていることを見逃す。"""
    markup = _overlay(overall=0.02, equal_weight=0.031)

    assert format_symmetry(0.02) in markup
    assert format_score(0.031) in markup
    assert format_score(0.02) not in markup


def test_only_the_score_goes_missing_when_a_single_part_is_missing() -> None:
    """§8.6.5/AC-SCORE-10: 部位が 1 つ欠測すると `s_3d` だけが欠測になり、
    `d_3d` は定義され続ける。吹き出しでもこの非対称な欠測条件を保つ。"""
    markup = _overlay(
        overall=0.018,
        equal_weight=None,
        parts={"contour": 0.01, "eyes": 0.02, "nose": None, "mouth": 0.015},
    )

    assert format_symmetry(0.018) in markup
    score_block = markup.split('class="fsl-still-score"', 1)[1].split("</div>", 1)[0]
    assert "—" in score_block


def test_every_reading_is_a_dash_when_the_three_dimensional_result_is_missing() -> None:
    """不変条件 11: §8.5.1 の重ね合わせが成立しないときは全部 `—`。0 や前値でない。"""
    markup = _overlay(overall=None, equal_weight=None, parts=dict.fromkeys(PART_NAMES))

    # stylesheet のコメントにも `—` は出るので、吹き出しの本体だけを数える。
    body = markup.split("</style>")[-1]
    assert body.count("—") == 1 + 1 + len(PART_NAMES)


def test_the_overlay_stylesheet_survives_the_render() -> None:
    """実ブラウザで見つけた 2 つの落とし穴を固定する（2026-08-12）。

    (a) **`st.html` は `<style>` を丸ごと落とす**（Streamlit 1.59.1 で実測）。
        markup と inline の `style` 属性だけが通るので、stylesheet が消えて
        吹き出しが `position: static` のまま画像の下へ流れ落ちた。
    (b) Streamlit は要素を `.stElementContainer`（`position: relative`）で包む。
        これを静的へ戻さないと、包含ブロックが画像の直後にある高さ 0 のラッパーに
        なり、吹き出しが画像の外へ出て、`max-height` の % も高さ 0 に対して解決
        されて中身が潰れた。

    どちらも「テストは通るのに画面には出ない」種類の壊れ方なので、markup の側で
    固定する。罠: `st.html` へ戻すと (a) で、neutraliser を消すと (b) で落ちる。
    """
    at = AppTest.from_string(_render_source())
    at.run()

    assert not at.exception
    markup = _rendered_html(at)
    assert "<style>" in markup
    assert f".st-key-{STILL_OVERLAY_CONTAINER_KEY} {{" in markup
    assert ".stElementContainer:has(.fsl-still-hud)" in markup
    assert "position: absolute" in markup
    assert "position: fixed" in markup  # ドラッグ後の `.fsl-still-floating`


def test_the_container_width_is_pinned_to_the_preview_width() -> None:
    """§5.1: コンテナは画像とちょうど同じ幅にする。

    CSS の `width: fit-content` では Streamlit の縦ブロックが伸ばしてしまい、
    実測でコンテナだけ 735px のまま残った（画像は 547px）。コンテナが画像より
    広いと `left: 18px` が画像の外を指すので、幅は Python 側で決めて注入する。
    罠: 動的な幅の注入を外すと、この assertion が落ちる。
    """
    from face_symmetry_lab.ui.still_overlay import preview_display_width

    at = AppTest.from_string(_render_source())
    at.run()

    assert not at.exception
    expected = preview_display_width(height=8, width=8)
    assert f".st-key-{STILL_OVERLAY_CONTAINER_KEY} {{ width: {expected}px; }}" in _rendered_html(at)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (480, 640, 640),  # 上限内。拡大も縮小もしない
        (4032, 3024, 390),  # 縦長: 高さ 520 で頭打ち → 3024 × (520/4032)
        (3000, 4000, 693),  # 横長の巨大: 幅 760 で頭打ち → 4000 × (520/3000) が先
        (100, 100, 100),  # 小さい写真を引き伸ばさない
    ],
)
def test_the_preview_is_capped_without_being_enlarged(
    height: int, width: int, expected: int
) -> None:
    """§5.1: 縦横比を保ったまま上限へ収め、**拡大はしない**。

    小さな写真を引き伸ばすと、ランドマークの点が実際より粗く見え、検出の精度に
    関する誤解を招く。
    """
    from face_symmetry_lab.ui.still_overlay import preview_display_width

    assert preview_display_width(height=height, width=width) == expected


# --- ドラッグ（§14.3e、2026-08-12） -------------------------------------------


def test_the_overlay_carries_a_grip_that_alone_receives_pointer_events() -> None:
    """§14.3e: ドラッグを受けるのはグリップだけ。本体は `pointer-events: none` の
    まま、画像の選択・右クリックを奪わない（HUD と同じ規約）。"""
    markup = _overlay()

    assert "fsl-still-grip" in markup
    assert "pointer-events: none" in markup
    assert "pointer-events: auto" in markup


def test_the_drag_runtime_delegates_instead_of_binding_to_the_element() -> None:
    """§14.3e: Streamlit は再実行のたびに吹き出しの DOM を作り直す。

    要素へ直接 listener を付けると、最初の再実行でドラッグが黙って死ぬ。
    罠: `grip.addEventListener(...)` を吹き出しの要素へ直接書いた実装は落ちる。
    """
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    assert "addEventListener('pointerdown'" in script
    assert "closest('.fsl-still-grip')" in script


def test_the_drag_runtime_clamps_to_the_viewport_not_the_image() -> None:
    """§14.3e（2026-08-12 改訂）: 移動範囲はページ全体。

    画像の内側だけに閉じ込めると、写真の全面に顔が写っている場合に退ける先が
    無くなる。HUD と同じく viewport に対して、自身の幅・高さぶんだけ右端・下端も
    制限する。罠: 包含ブロック（`offsetParent`）へ clamp する実装は、
    `innerWidth` を使わないのでここで落ちる。
    """
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    assert "innerWidth" in script
    assert "innerHeight" in script
    assert "offsetWidth" in script
    assert "offsetHeight" in script
    assert "offsetParent" not in script


def test_the_overlay_is_visible_without_any_javascript() -> None:
    """実障害の回帰（2026-08-12）: 画像アップロード後に吹き出しが見えなくなった。

    途中の版は吹き出しを `display: none` の source として置き、画面へ出すのを
    bridge の仕事にしていた。**bridge が動かない環境では吹き出しが丸ごと消える。**
    表示は CSS の責任、移動は JS の上乗せ、と分ける。罠: markup を隠す実装や、
    JS が作る要素へ移す実装はここで落ちる。
    """
    markup = _overlay()

    assert 'class="fsl-still-hud"' in markup
    assert "hidden" not in markup
    assert "display: none" not in markup.split("</style>")[-1]
    assert "position: absolute" in markup


def test_the_drag_runtime_never_destroys_streamlit_owned_dom() -> None:
    """実障害の回帰（2026-08-12）: 画像アップロード後に吹き出しが完全に消えた。

    最初の実装は Streamlit が描いた吹き出しの要素そのものを `<body>` へ移し、
    プレビューのコンテナが見つからないときは `remove()` していた。**再描画の
    途中ではコンテナが一瞬だけ消える**ため、そこで描画済みの吹き出しごと巻き
    添えに消え、Streamlit 側は「その要素はもう描いてある」と思っているので二度と
    描き直さなかった。

    したがって runtime は Streamlit の DOM を読むだけにする。source が無い間は
    自前のホストを隠すだけで、何も削除しない。罠: `remove()` を書き戻すと落ちる。
    """
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    assert ".remove()" not in script.split("<script>")[-1].split("var previous")[0]
    assert "appendChild(fresh" not in script
    assert "innerHTML" not in script


def test_the_drag_promotes_from_the_current_on_screen_position() -> None:
    """§14.3e: 初期位置は CSS が決める画像の左上。掴んだ瞬間に飛ばないよう、
    最初のドラッグでいまの見た目の位置をそのまま viewport 座標へ写し取る。"""
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    assert "getBoundingClientRect" in script
    assert "classList.contains" in script


def test_the_drag_runtime_only_toggles_a_class_and_coordinates() -> None:
    """§14.3e: bridge の仕事は class と `left`/`top` の書き換えだけ。

    要素の生成・移動・削除をしないので、Streamlit の再描画と競合しない。
    Camera へ切り替えれば Streamlit が画像セクションごと消すため、掃除も要らない。
    """
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    assert "fsl-still-floating" in script
    assert "classList.add" in script
    assert "createElement('div')" not in script


def test_the_drag_runtime_keeps_no_storage() -> None:
    """§14.3e/§3.4: 位置は page realm の変数だけに持ち、永続化しない。"""
    from face_symmetry_lab.ui.still_overlay import still_drag_bridge_script

    script = still_drag_bridge_script()

    for storage in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert storage not in script


def test_the_drag_runtime_is_installed_once_per_render() -> None:
    """bridge は 1 回の描画につき 1 つだけ。積み増すと listener が二重に付く。"""
    at = AppTest.from_string(_render_source())
    at.run()

    assert not at.exception
    iframes = [element for element in at.get("iframe")]
    assert len(iframes) == 1
