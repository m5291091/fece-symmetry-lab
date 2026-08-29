"""英語の訳表: `ui/hud.py`, `ui/hud_assets.py`, `ui/still_overlay.py`（FR-043）。

**`ui.hud.HUD_POSITION_OPTIONS`（"左上"/"右上"/"左下"/"右下"）と
`ui.hud.HUD_HIDDEN`（"非表示"）の 5 語は、ここに訳を用意するだけである。**
これらは `st.radio` の選択肢であると同時に `camera_hud_position` という
session state の値そのものであり、`ui.hud.HUD_PLACEMENTS`（CSS の
`top/left/right/bottom` 指定）を引く鍵でもある。訳すと言語切替の瞬間に
保存済みの選択が options から外れ、`HUD_PLACEMENTS` の引き当ても壊れる。
`ui.hud` 側は識別子のまま `options=` へ渡し、表示だけを訳す
`format_func=lambda v: t(v)` は `ui.camera`（別担当）が付ける。したがって
この 5 語は、`t(...)` の静的呼び出しからまだ参照されない
（`ui.camera` 側の実装が入るまでの間、一時的に「孤児」になる）。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # `ui.hud.HUD_POSITION_OPTIONS` / `HUD_HIDDEN`。上の module docstring を参照。
    "左上": "Top left",
    "右上": "Top right",
    "左下": "Bottom left",
    "右下": "Bottom right",
    "非表示": "Hidden",
    # `ui.hud.build_hud_payload` の "label"。
    "対称度（実験値）": "Symmetry (experimental)",
    # `ui.hud._HUD_AGE_TEMPLATE` / `_HUD_DEMOGRAPHIC_TEMPLATE`。
    # 元は f-string だったため、テンプレートへ切り出してから訳した
    # （§3b〜3f「包み方の規約」2）。HUD の吹き出しは幅が狭いので短く保つ。
    "{age:.0f} 歳": "{age:.0f} years",
    "参考年齢 {age} · 参考性別 {gender}": "Age {age} · Gender {gender}",
    # `ui.hud.FULLSCREEN_BADGE` / `FULLSCREEN_HINT`。全画面中に残る唯一の文章。
    "全画面表示中 — 右上の「終了」で戻ります": "Full screen — use Exit (top right) to return",
    "画面下の「START」を押すとカメラが始まります。"
    "ブラウザの F11 と併用すると、タブやアドレスバーも隠れた完全な全画面になります。": (
        "Press START at the bottom to start the camera. Combine with your "
        "browser's F11 for a full screen without tabs or the address bar."
    ),
    # `ui.hud.hud_bridge_script` が CFG へ載せ、`ui.hud_assets.HUD_RUNTIME_JS`
    # が textContent として書き込む部位ラベル・凡例・スコア見出し・grip 文言。
    # 部位名は仕様どおり固定する。
    "全体のスコア": "Overall score",
    "輪郭": "Contour",
    "目": "Eyes",
    "鼻": "Nose",
    "口": "Mouth",
    "全体": "Overall",
    # grip の文言。「⠿」は装飾記号なので訳の対象に含めない
    # （`ui.hud_assets` の CFG.gripText と `ui.still_overlay._DRAG_GRIP_LABEL`
    # の両方から参照される、対になった 1 語）。
    "ドラッグで移動": "Drag to move",
}
