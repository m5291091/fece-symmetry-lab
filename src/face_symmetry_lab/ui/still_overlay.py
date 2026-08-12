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
2. **移動機能を持たない。** 配置プリセットもドラッグも持たない。1 枚に対する
   1 回きりの表示であり、映像のように下にある `STOP` を塞ぐ問題も起きない。
3. **JS bridge を使わない。** 5Hz の payload も page realm への runtime 注入も
   無く、`st.container(key=...)` が作る位置指定コンテキストの中へ静的な markup
   を重ねるだけである。bridge が要ったのは 5Hz 更新と擬似全画面という、静止画に
   無い 2 つの要件のためだった。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import streamlit as st
from numpy.typing import NDArray

from face_symmetry_lab.domain.metrics import PART_NAMES
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.ui.hud import build_hud_payload
from face_symmetry_lab.ui.scores import PART_EQUAL_WEIGHT_LABEL, PART_LABELS

#: 位置指定コンテキストを作るコンテナの key。`st.container(key=...)` は
#: `.st-key-<key>` という class を吐くので、構造依存セレクタ（`nth-child` 等）
#: を使わずに CSS から画像の枠を特定できる。HUD が
#: `.st-key-camera_preview_frame` を使うのと同じ理由であり、Streamlit の内部
#: 構造が変わっても黙って壊れない。
STILL_OVERLAY_CONTAINER_KEY: Final[str] = "image_preview_frame"

#: 吹き出しの見た目。HUD（`ui.hud.HUD_STYLESHEET`）と同じ語彙にそろえてある
#: ——半透明の黒地、白文字、角丸、ゲート色の枠、tabular-nums。`position:
#: absolute` の基準は上のキー付きコンテナである。
#:
#: `pointer-events: none` は HUD と同じ理由で残す。静止画には塞ぐ操作が無いが、
#: 画像そのものの右クリック・選択を吹き出しが奪わないほうが素直である。
_STILL_OVERLAY_CSS: Final[str] = f"""
.st-key-{STILL_OVERLAY_CONTAINER_KEY} {{ position: relative; }}
/* Streamlit は要素 1 つずつを `.stElementContainer` で包み、それへ
   `position: relative` を付ける。**この中間ラッパーを静的へ戻さないと、
   吹き出しの包含ブロックがキー付きコンテナではなくラッパーになる。** ラッパーは
   画像の直後に高さ 0 で置かれるので、`top: 18px` が画像の下端 + 18px を指し、
   吹き出しが画像の外へ落ちる。さらに `max-height` の % が高さ 0 に対して解決され、
   中身が 32px へ潰れる（実ブラウザで両方とも確認した実際の症状）。
   `:has()` を使うのは、この stylesheet 自身がラッパーの内側へ出力されるため、
   親を選ぶ手段が他に無いからである。子ではなく子孫で照合し、Streamlit が
   ラッパーを 1 枚増やしても壊れないようにする。 */
.st-key-{STILL_OVERLAY_CONTAINER_KEY} .stElementContainer:has(.fsl-still-hud) {{
    position: static;
}}
.st-key-{STILL_OVERLAY_CONTAINER_KEY} .fsl-still-hud {{
    position: absolute; top: 18px; left: 18px; z-index: 10;
    background: rgba(0,0,0,.72); color: #fff;
    padding: 14px 20px; border-radius: 16px;
    pointer-events: none;
    font-variant-numeric: tabular-nums;
    font-family: inherit;
    max-width: min(340px, 76%); max-height: 88%; overflow-y: auto;
}}
.fsl-still-value {{ font-size: 2.6rem; font-weight: 700; line-height: 1; }}
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
"""


def still_overlay_html(
    *,
    overall: float | None,
    equal_weight: float | None,
    parts: Mapping[str, float | None],
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
) -> str:
    """吹き出し 1 つ分の markup を組み立てる純関数。

    数値は `build_hud_payload` が返した表示文字列をそのまま置くだけで、ここでは
    一切整形しない。`position` と `generation` は HUD の移動と折れ線グラフの
    世代管理のための引数なので、静止画では固定値を渡して結果を捨てる。

    参考年齢・参考性別は `fsl-still-demo` という**独立した要素**へ入れる。
    左右差と同じ行へ混ぜないのは不変条件 5（§1.3.1）が求めるところであり、
    推定が無いときや利用者が切ったときは行ごと出さない（空行は「動いていて
    失敗している」と読まれる、FR-024）。
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
    )
    readings = payload["parts"]
    assert isinstance(readings, Mapping)
    part_items = "".join(
        f'<span class="fsl-still-part"><b>{PART_LABELS[name]}</b>{readings[name]}</span>'
        for name in PART_NAMES
    )
    demo = payload["demo"]
    demo_row = f'<div class="fsl-still-demo">{demo}</div>' if demo else ""
    return (
        f"<style>{_STILL_OVERLAY_CSS}</style>"
        f'<div class="fsl-still-hud" style="border: 2px solid {gate};">'
        f'<div class="fsl-still-value">{payload["symmetry"]}</div>'
        f'<div class="fsl-still-label">{payload["label"]}</div>'
        f'<div class="fsl-still-score">'
        f"<b>{PART_EQUAL_WEIGHT_LABEL}</b>{payload['score']}</div>"
        f'<div class="fsl-still-parts">{part_items}</div>'
        f"{demo_row}"
        "</div>"
    )


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

    with st.container(key=STILL_OVERLAY_CONTAINER_KEY):
        st.image(array, caption=caption, width="stretch")
        st.markdown(
            still_overlay_html(
                overall=overall,
                equal_weight=equal_weight,
                parts=parts,
                estimate=estimate,
                show_demographics=show_demographics,
                gate=gate,
            ),
            unsafe_allow_html=True,
        )
