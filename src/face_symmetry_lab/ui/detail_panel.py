"""§5.1: 「詳細結果」タブの枠を Camera と Image で 1 つにする。

数値そのものは `ui.scores.render_score_details` が描く。ここが持つのは**その
まわりの形**——囲み・回復導線・何も出せないときの 1 行・実行時の数値を入れる
折りたたみ——だけである。

**分離した理由は、同じ性質の情報が入力モードで違う場所に出ていたことである。**
Camera は実行時の数値（フレーム統計・処理時間）を結果の下の折りたたみへ入れて
いたのに対し、Image は入力サイズとランドマーク点数を測定結果より**上**に素の
caption として置いていた。どちらも「利用者が手を打てない診断値」であり、
測定結果より先に読ませる理由が無い（§14.3f）。

``show_parts`` を残しているのは AC-LAYOUT-01 が入力ごとに違う値を要求するため
である——Camera は映像上の HUD が既に部位別 4 つを出すので詳細タブでは省き、
Image の詳細タブには部位別 4 つが残る。**枠を共有してもこの差は握りつぶさない。**
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import streamlit as st

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.ui.scores import render_score_details

#: §14.3f: 実行時の数値を入れる折りたたみの見出し。Camera と Image で同じ語に
#: する（同じ性質の情報が、入力モードで違う名前を持たないように）。
DIAGNOSTICS_LABEL: Final[str] = "処理の状態（診断用）"


def render_detail_panel(
    outcome: AnalysisOutcome | None,
    *,
    empty_notice: str,
    show_parts: bool,
    warning: str | None = None,
    diagnostics: Callable[[], None] | None = None,
) -> None:
    """詳細結果を、どちらの入力でも同じ形・同じ順で描く。

    順序は 1 つに固定する。**回復導線（あれば）→ 測定結果か、出せない理由
    → 診断用の折りたたみ（あれば）**。測定結果を先頭付近に置くのは、利用者が
    手を打てない実行時の数値で結果を埋もれさせないためである。

    ``show_smoothed`` は渡さない（常に生値を主表示する）。静止画は平滑値を持たず、
    Camera の詳細タブは HUD が出す平滑値を重複表示しないため、どちらの入力でも
    `False` が正しい——ここで分岐させると、同じ測定値に 2 つの表示規則ができる。
    """
    with st.container(border=True):
        if warning:
            st.warning(warning)

        if outcome is None:
            st.info(empty_notice)
        else:
            render_score_details(outcome, show_smoothed=False, show_parts=show_parts)

        if diagnostics is not None:
            st.divider()
            with st.expander(DIAGNOSTICS_LABEL):
                diagnostics()
