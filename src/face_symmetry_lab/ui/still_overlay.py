"""静止画に重ねる吹き出し（§5.1・§14.3e、2026-08-12、AC-LAYOUT-01）。

Camera が映像上の HUD で行っていることを、Image でも同じ見た目・同じ数値で
行う。これにより Image 側の見出し結果（画像の隣に並べていた数値の区画）を
削除でき、Camera が項目 26（H3）で解消したのと同じ「同じ測定値が 2 か所へ出る」
状態を Image でも作らずに済む。

**表示文字列は `ui.hud.build_hud_payload` から取る。** ここで整形し直さない。
同じ測定値に対して Camera と Image が二通りの表示規則へ分岐しないための境界で
あり、`ui.hud` 側のテストが守っている不変（共有フォーマッタを通すこと・欠測は
`—`）がそのままここにも効く。

**HUD と違うのは 3 点だけである。**

1. **折れ線グラフを持たない。** 静止画は時系列を持たないので、描く系列が存在
   しない。リングバッファも世代識別子も要らない。
2. **配置プリセットを持たない。** ドラッグ（§14.3e、2026-08-12 追加）だけを持つ。
   1 枚に対する 1 回きりの表示なので、直接動かせれば足りる。位置は page realm の
   変数にだけ持ち、`localStorage` へ保存しない。
3. **JS bridge はドラッグのためだけに使う。** 5Hz の payload もグラフの runtime も
   持たず、親 realm へ入れるのはイベント委譲 1 つだけである。bridge が HUD で
   大きくなったのは 5Hz 更新と擬似全画面という、静止画に無い 2 つの要件のため
   だった。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from numpy.typing import NDArray

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.ui.hud import build_hud_payload
from face_symmetry_lab.ui.readings import PART_LABELS
from face_symmetry_lab.ui.scores import PART_EQUAL_WEIGHT_LABEL

#: 位置指定コンテキストを作るコンテナの key。`st.container(key=...)` は
#: `.st-key-<key>` という class を吐くので、構造依存セレクタ（`nth-child` 等）
#: を使わずに CSS から画像の枠を特定できる。HUD が
#: `.st-key-camera_preview_frame` を使うのと同じ理由であり、Streamlit の内部
#: 構造が変わっても黙って壊れない。
STILL_OVERLAY_CONTAINER_KEY: Final[str] = "image_preview_frame"

#: プレビューの表示上限（§5.1）。`st.image` の既定は幅いっぱいなので、縦長の
#: 写真（スマートフォンの 3:4 など）ではページが画像だけで埋まり、詳細結果が
#: 折り返しの遥か下へ行く。高さ側にも上限を置くことで、縦横どちらの写真でも
#: 画像と詳細結果の先頭が同じ画面に収まる。
MAX_PREVIEW_WIDTH_PX: Final[int] = 760
MAX_PREVIEW_HEIGHT_PX: Final[int] = 520


def preview_display_width(*, height: int, width: int) -> int:
    """縦横比を保ったまま上限へ収めたときの、表示幅（px）。

    **拡大はしない**（`scale` の上限は 1.0）。小さな写真を引き伸ばすと、
    ランドマークの点が実際より粗く見え、検出の精度に関する誤解を招く。

    この値を `st.image` の幅としても、コンテナの幅としても使う。両者が同じ数から
    来ることが、吹き出しの `left` / `top` が画像の内側を指す条件である。
    """
    if height <= 0 or width <= 0:
        raise ValueError("プレビューの寸法は正の値である必要があります。")
    scale = min(MAX_PREVIEW_WIDTH_PX / width, MAX_PREVIEW_HEIGHT_PX / height, 1.0)
    return max(1, round(width * scale))


#: 吹き出しの見た目。HUD（`ui.hud.HUD_STYLESHEET`）と同じ語彙にそろえてある
#: ——半透明の黒地、白文字、角丸、ゲート色の枠、tabular-nums。`position:
#: absolute` の基準は上のキー付きコンテナである。
#:
#: `pointer-events: none` は HUD と同じ理由で残す。静止画には塞ぐ操作が無いが、
#: 画像そのものの右クリック・選択を吹き出しが奪わないほうが素直である。
_STILL_OVERLAY_CSS: Final[str] = f"""
/* コンテナは画像とちょうど同じ幅にする（幅は `preview_display_width` が Python
   側で決め、下の動的な 1 行で注入する）。**CSS の `width: fit-content` では
   縮まない** —— Streamlit の縦ブロックは flex で伸ばされるため、実測でコンテナ
   だけ 735px のまま残り、中央寄せされた画像の外側に吹き出しが付いた。
   コンテナが画像より広いと、`left: 18px` が画像の外を指す。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} {{
    position: relative; max-width: 100%; margin-inline: auto;
}}
/* `st.image(..., width=<px>)` は `<img>` へ固定幅を書き込む。上限（760px）より
   狭いブラウザ幅では、コンテナ（`max-width: 100%`）だけが縮んで画像がその外へ
   はみ出し、吹き出しの基準と画像の左端がずれる。縦横比を保ったまま一緒に縮む
   ようにしておく。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} img {{
    display: block; max-width: 100%; height: auto;
}}
/* Streamlit は要素 1 つずつを `.stElementContainer`（`position: relative`）で
   包む。これを静的へ戻さないと、吹き出しの包含ブロックがキー付きコンテナでは
   なく、画像の直後にある高さ 0 のラッパーになる（実ブラウザで確認した症状）。
   `:has()` を使うのは、この stylesheet 自身がラッパーの内側へ出力されるため、
   親を選ぶ手段が他に無いからである。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} .stElementContainer:has(.fsl-still-hud) {{
    position: static;
}}
/* **既定では JS を一切必要とせず、CSS だけで画像の上に見える。**
   ここを JS 依存（隠しておいて JS が出す）にした版で、bridge が動かない環境では
   吹き出しが丸ごと消えた。表示は CSS の責任、移動は JS の上乗せ、と分ける。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} .fsl-still-hud {{
    position: absolute; top: 18px; left: 18px; z-index: 1000050;
    background: rgba(0,0,0,.72); color: #fff;
    padding: 14px 20px; border-radius: 16px;
    pointer-events: none;
    font-variant-numeric: tabular-nums;
    font-family: inherit;
    /* 縦長の写真（上限 520px 高 = 幅 390px 前後）で 76% を占めて顔を隠したため、
       割合側を絞った。ドラッグで退けられるとはいえ、既定で顔を覆わない。 */
    max-width: min(320px, 64%); max-height: 88%; overflow-y: auto;
}}
/* ドラッグが始まると bridge がこの class を付け、viewport 基準へ切り替える。
   要素は Streamlit が置いた場所に留まる（動かすと再描画と競合する）。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} .fsl-still-hud.fsl-still-floating {{
    position: fixed;
    max-width: min(320px, 88vw); max-height: 88vh;
}}
.fsl-still-value {{ font-size: 2.2rem; font-weight: 700; line-height: 1; }}
.fsl-still-label {{ font-size: .8rem; opacity: .75; margin-bottom: 10px; }}
.fsl-still-score {{
    display: flex; align-items: baseline; gap: 8px;
    font-size: 1.3rem; font-weight: 600; margin-top: 2px;
}}
.fsl-still-score b {{ font-size: .78rem; font-weight: 400; opacity: .75; }}
.fsl-still-parts {{
    display: flex; flex-wrap: wrap; gap: 4px 14px;
    font-size: .92rem; margin-top: 8px;
}}
.fsl-still-parts b {{ font-weight: 400; opacity: .75; margin-right: 4px; }}
.fsl-still-demo {{ font-size: 1.05rem; margin-top: 8px; }}
/* ドラッグを受けるのはグリップだけ。吹き出し本体は `pointer-events: none` の
   ままにして、下にある画像の選択・右クリックを奪わない（HUD と同じ規約）。 */
.fsl-still-grip {{
    pointer-events: auto; cursor: move; touch-action: none;
    margin-top: 10px; font-size: .72rem; opacity: .55;
    border-top: 1px solid rgba(255,255,255,.25); padding-top: 6px;
}}
"""

#: 親 realm へ入れるドラッグの実行時コード。**イベント委譲にする。**
#: Streamlit は再実行のたびに吹き出しの DOM を作り直すので、要素へ直接 listener
#: を付けると最初の再実行でドラッグが黙って死ぬ。`document` 上で拾えば、DOM が
#: 何度作り直されても効き続ける。
#:
#: 位置は `window.__fslStillPos` にだけ持ち、storage へ書かない（§14.3e）。
#: 写真ごとに最適な退避先が変わるため前回位置を復元する利点が乏しく、保存しな
#: ければ §3.4 の永続化の議論そのものが発生しない。
_STILL_DRAG_JS: Final[str] = """
(function () {
  // **Streamlit が所有する DOM を、動かしも消しもしない。** 途中の版は要素を
  // `<body>` へ移し、コンテナが見つからないときは `remove()` していた。再描画の
  // 途中ではコンテナが一瞬だけ消えるため、そこで描画済みの吹き出しごと巻き添えに
  // 消え、Streamlit は「もう描いてある」と判断して二度と描き直さなかった。
  //
  // **表示は CSS の責任で、JS は移動だけを上乗せする。** 隠しておいて JS が出す
  // 形にした版では、bridge が動かない環境で吹き出しが丸ごと見えなくなった。
  // ここで行うのは class と `left`/`top` の書き換えだけである。
  var FLOATING = 'fsl-still-floating';

  var place = function (hud, x, y) {
    hud.classList.add(FLOATING);
    hud.style.left = x + 'px';
    hud.style.top = y + 'px';
    hud.style.right = 'auto';
    hud.style.bottom = 'auto';
    window.__fslStillPos = {x: x, y: y};
  };

  // 既定は CSS の `absolute`（画像の内側）。最初のドラッグでいまの見た目の位置を
  // そのまま viewport 座標へ写し取り、`fixed` へ切り替える。こうすると掴んだ
  // 瞬間に飛ばない。
  var promote = function (hud) {
    if (hud.classList.contains(FLOATING)) return;
    var box = hud.getBoundingClientRect();
    place(hud, box.left, box.top);
  };

  // 再描画で作り直された吹き出しへ、直前の位置を貼り直す。
  var restore = function () {
    var saved = window.__fslStillPos;
    if (!saved) return;
    var hud = document.querySelector('.fsl-still-hud');
    if (hud && !hud.classList.contains(FLOATING)) place(hud, saved.x, saved.y);
  };
  new MutationObserver(restore).observe(document.body, {childList: true, subtree: true});
  restore();

  if (window.__fslStillDrag) return;
  window.__fslStillDrag = true;

  var dragging = null, offsetX = 0, offsetY = 0;

  document.addEventListener('pointerdown', function (event) {
    var grip = event.target.closest && event.target.closest('.fsl-still-grip');
    if (!grip) return;
    var hud = grip.closest('.fsl-still-hud');
    if (!hud) return;
    promote(hud);
    var box = hud.getBoundingClientRect();
    dragging = hud;
    offsetX = event.clientX - box.left;
    offsetY = event.clientY - box.top;
    try { grip.setPointerCapture(event.pointerId); } catch (err) {}
    event.preventDefault();
  });

  document.addEventListener('pointermove', function (event) {
    if (!dragging) return;
    // HUD と同じ viewport への clamp。自身の幅・高さぶんだけ右端・下端も
    // 制限し、画面外へ完全に出て操作不能になることを防ぐ。
    var maxX = Math.max(0, window.innerWidth - dragging.offsetWidth);
    var maxY = Math.max(0, window.innerHeight - dragging.offsetHeight);
    var x = Math.min(maxX, Math.max(0, event.clientX - offsetX));
    var y = Math.min(maxY, Math.max(0, event.clientY - offsetY));
    place(dragging, x, y);
  });

  var stop = function () { dragging = null; };
  document.addEventListener('pointerup', stop);
  document.addEventListener('pointercancel', stop);
})();
""".replace("__CONTAINER_KEY__", STILL_OVERLAY_CONTAINER_KEY)


def still_overlay_html(
    *,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    notice: str = "",
) -> str:
    """吹き出し 1 つ分の markup を組み立てる純関数。

    数値は `build_hud_payload` が返した表示文字列をそのまま置くだけで、ここでは
    一切整形しない。`position` と `generation` は HUD の移動と折れ線グラフの
    世代管理のための引数なので、静止画では固定値を渡して結果を捨てる。

    参考年齢・参考性別は `fsl-still-demo` という**独立した要素**へ入れる。
    左右差と同じ行へ混ぜないのは不変条件 5（§1.3.1）が求めるところであり、
    推定が無いときや利用者が切ったときは行ごと出さない（空行は「動いていて
    失敗している」と読まれる、FR-024）。``notice``（FR-038）は
    `show_demographics` が偽のときにだけ、その独立した行の中身を年齢ゲートの
    固定文言（`ui.age_gate_notice`）へ差し替える。
    """
    payload = build_hud_payload(
        overall=overall,
        equal_weight=equal_weight,
        parts=parts,
        estimate=estimate,
        show_demographics=show_demographics,
        gate=gate,
        position="左上",
        generation="",
        notice=notice,
    )
    readings = payload["parts"]
    assert isinstance(readings, Mapping)
    part_items = "".join(
        f'<span class="fsl-still-part"><b>{PART_LABELS[name]}</b>{readings[name]}</span>'
        for name in PART_NAMES
    )
    demo = payload["demo"]
    demo_row = f'<div class="fsl-still-demo">{demo}</div>' if demo else ""
    grip = '<div class="fsl-still-grip">⠿ ドラッグで移動</div>'
    return (
        f"<style>{_STILL_OVERLAY_CSS}</style>"
        # **この要素がそのまま画面に出る。** JS が動かなくても CSS だけで画像の
        # 上に見える（`_STILL_OVERLAY_CSS` を参照）。bridge は class と座標を
        # 書き換えるだけで、この要素を動かしも消しもしない。
        f'<div class="fsl-still-hud" style="border: 2px solid {gate};">'
        f'<div class="fsl-still-value">{payload["symmetry"]}</div>'
        f'<div class="fsl-still-label">{payload["label"]}</div>'
        f'<div class="fsl-still-score">'
        f"<b>{PART_EQUAL_WEIGHT_LABEL}</b>{payload['score']}</div>"
        f'<div class="fsl-still-parts">{part_items}</div>'
        f"{demo_row}"
        f"{grip}"
        "</div>"
    )


def still_drag_bridge_script() -> str:
    """`_STILL_DRAG_JS` を親 realm へ設置するローダー。

    `ui.hud.hud_bridge_script` と同じ形である。`components.v1.html` の iframe は
    ページと same-origin なので `window.parent` へ届く。iframe 内で定義したものは
    次の再実行で Streamlit がそれを差し替えた瞬間に動かなくなるため、iframe 側の
    仕事は実行時コードを親の `<head>` へ渡すことだけに留める。

    **HUD の bridge とは別の要素 id を使う。** 同じ id にすると、Camera から
    Image へ切り替えたときに一方が他方を消してしまう。
    """
    runtime = json.dumps(_STILL_DRAG_JS)
    return f"""
<script>
(function () {{
  var win = window.parent;
  var doc = win && win.document;
  if (!doc || !doc.body) return;
  var previous = doc.getElementById('fsl-still-drag-runtime');
  if (previous) previous.remove();
  var runtime = doc.createElement('script');
  runtime.id = 'fsl-still-drag-runtime';
  runtime.textContent = {runtime};
  doc.head.appendChild(runtime);
}})();
</script>
"""


def render_image_with_overlay(
    preview: NDArray[np.uint8],
    *,
    caption: str,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    notice: str = "",
) -> None:
    """プレビュー画像と、その上に重ねる吹き出しを描く。

    画像は `st.image` がそのまま描く（pixel へ数値を焼き込まない、FR-013）。
    吹き出しは同じキー付きコンテナの中へ置き、CSS の `position: absolute` で
    画像の上へ重ねる。**画像を再エンコードしない。**

    **`st.html` ではなく `st.markdown(unsafe_allow_html=True)` を使う。**
    `st.html` は `<style>` 要素を丸ごと落とし、markup と inline の `style` 属性
    だけを通す（Streamlit 1.59.1 で実測）。stylesheet が消えると吹き出しは
    `position: static` のまま画像の下へ流れ落ちるので、この経路は使えない。
    """
    array = np.asarray(preview)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("吹き出しを重ねる画像は RGB の 3 次元配列である必要があります。")

    display_width = preview_display_width(height=array.shape[0], width=array.shape[1])
    with st.container(key=STILL_OVERLAY_CONTAINER_KEY):
        st.image(array, caption=caption, width=display_width)
        st.markdown(
            f"<style>.st-key-{STILL_OVERLAY_CONTAINER_KEY} {{ width: {display_width}px; }}</style>",
            unsafe_allow_html=True,
        )
        st.markdown(
            still_overlay_html(
                overall=overall,
                equal_weight=equal_weight,
                parts=parts,
                estimate=estimate,
                show_demographics=show_demographics,
                gate=gate,
                notice=notice,
            ),
            unsafe_allow_html=True,
        )
        # ドラッグの実行時コードは、吹き出しと同じコンテナの中へ 1 つだけ置く
        # （`height=0` なので場所は取らない）。積み増すと listener が二重に付く。
        components.html(still_drag_bridge_script(), height=0)
