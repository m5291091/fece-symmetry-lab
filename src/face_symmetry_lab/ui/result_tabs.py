"""映像・画像の下に並ぶタブの名前・順序・アイコンを 1 か所で持つ（§5.1）。

**分離した理由は食い違いである。** `ui.camera` と `ui.image_section` はそれぞれ
自前で `st.tabs([...])` を書いており、共通のはずのラベルが実際にずれていた
（Camera は「まとめ・エクスポート」、Image は「エクスポート」）。§5.1 は
「Camera と Image は、映像・画像の下に並ぶタブの名前・順序・アイコンをそろえる」
と定めているので、**そろっていることを規約ではなく型で保つ**——ラベル列と中身の
対応をこの module だけが決め、呼び出し側はどのタブへ何を描くかだけを渡す。

Camera と Image で違うのは「表示設定」を持つかどうかだけである（Image に映像の
見え方・HUD の位置・平滑化に相当する設定は無い）。その差は `ResultTabs.settings`
が `None` かどうかで表し、**位置は呼び出し側に選ばせない**——参考年齢・参考性別と
レポートの間、と決めておかないと、入力を切り替えたときにレポートの位置が動く。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import streamlit as st

#: アイコンは並んだタブを見分ける手掛かりであって、文言の置き換えではない
#: （§5.4: 記号だけで意味を伝えない）。達成度・良し悪しを表す絵文字は使わない
#: （§5.2）。
DETAIL_TAB_LABEL: Final[str] = "📊 詳細結果"
SYMMETRY_TAB_LABEL: Final[str] = "🪞 左右対称との比較"
DEMOGRAPHIC_TAB_LABEL: Final[str] = "👤 参考年齢・参考性別"
SETTINGS_TAB_LABEL: Final[str] = "⚙️ 表示設定"
#: §5.1 が共通のタブとして名指しする「レポート」。FR-042 が CSV / JSON を撤回
#: して以降、どちらの入力でも書き出せるのは HTML レポート 1 種類だけである。
#: Camera はその中身（FR-021 の要約）を画面にも出すが、**タブの名前は同じにする**
#: ——入力を切り替えたときに同じものを探す先が動かないようにするためである。
REPORT_TAB_LABEL: Final[str] = "📄 レポート"

#: タブ 1 つ分の中身。引数を取らないのは、呼び出し側が pipeline や解析結果を
#: closure で閉じ込めるためである（この module はどちらも知らない）。
TabBody = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ResultTabs:
    """映像・画像の下へ並べる中身。``settings`` を持つのは Camera だけである。"""

    detail: TabBody
    symmetry: TabBody
    demographic: TabBody
    report: TabBody
    settings: TabBody | None = None


def result_tab_labels(*, with_settings: bool) -> tuple[str, ...]:
    """§5.1 の並び。「表示設定」は参考年齢・参考性別とレポートの間に入る。"""
    settings = (SETTINGS_TAB_LABEL,) if with_settings else ()
    return (
        DETAIL_TAB_LABEL,
        SYMMETRY_TAB_LABEL,
        DEMOGRAPHIC_TAB_LABEL,
        *settings,
        REPORT_TAB_LABEL,
    )


def render_result_tabs(tabs: ResultTabs) -> None:
    """§5.1/AC-LAYOUT-01: 画面幅いっぱいで一度に 1 つだけ表示する。

    `st.tabs` はどのタブが表示されていても毎回すべての中身を構築する。したがって
    ここでの反復は「描く順」ではなく「どのラベルにどの中身が入るか」を決めている
    ——ラベル列と中身の列を同じ場所で組み立て、`strict=True` で長さのずれを
    その場で落とす。
    """
    bodies: tuple[TabBody, ...] = (
        tabs.detail,
        tabs.symmetry,
        tabs.demographic,
        *(() if tabs.settings is None else (tabs.settings,)),
        tabs.report,
    )
    labels = result_tab_labels(with_settings=tabs.settings is not None)
    for container, body in zip(st.tabs(list(labels)), bodies, strict=True):
        with container:
            body()
