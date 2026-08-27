"""FR-022/FR-024/§5.1: 参考年齢・参考性別の独立区画を Camera と Image で 1 つにする。

**分離した理由は状態の並べ方の食い違いである。** 以前は `ui.camera` と
`ui.image_section` が同じ区画を別々に書いており、「出せない理由」を見る順序が
違っていた——Camera はモデル未配置を最初に見るのに対し、Image は最後の
フォールバックとして見ていたため、モデルが無い環境の写真では「単一顔を検出できた
写真でのみ…」という、利用者に手の打ちようがない理由が出ていた。

**黙って消えない**という契約（FR-024、§5.1）はどちらの入力でも同じであり、
2 か所に置くと片方だけが分岐を落としても「動いているように見える」。表示可否の
判定そのものは `ui.age_gate_notice.resolve_reference_age_view` が持ち、ここは
その結果と「出せないときに何と言うか」だけを組み立てる。

入力モードで変わるのは 2 つの文だけである。**まだ推定できていないときの 1 行**
（Camera は「顔が写っている間に推定します」、Image は「単一顔を検出できた写真
でのみ」）と、**末尾へ添える 1 行**（Camera は §9.6.1 のセッション範囲、Image は
添えるものを持たない）。どちらも呼び出し側から渡す。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.application.age_gate import AgeGateOutcome
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.ui.age_gate_notice import resolve_reference_age_view
from face_symmetry_lab.ui.scores import DEMOGRAPHIC_RUNTIME_CAPTION, render_demographic_estimate

#: モデル資産が無い環境。**最初に見る**——他のどの理由を述べても利用者は手を
#: 打てないためである（FR-024 の「年齢・性別区画だけに回復手順を示す」）。
MODEL_MISSING_NOTICE: Final[str] = "年齢・性別モデルが未配置のため、参考値を表示できません。"

#: 利用者が自分で切っただけの状態。**ゲートの固定文言をここへ出さない**——
#: 切っただけの状態を「表示できません」と説明すると、機能が失敗しているように
#: 読める（FR-024）。Image はこの状態を持たない（表示設定を持たないため）。
DISPLAY_OFF_NOTICE: Final[str] = (
    "「表示設定」の「年齢・性別の参考値を表示」を入れると、ここに表示します。"
)


def render_demographic_panel(
    *,
    estimate: DemographicEstimate | None,
    gate_outcome: AgeGateOutcome | None,
    available: bool,
    pending_notice: str,
    show_demographics: bool = True,
    error: str | None = None,
    footnote: str | None = None,
) -> None:
    """参考年齢・参考性別を、どちらの入力でも同じ形・同じ順で描く。

    順序は 1 つに固定する。**来歴 → 出せない理由（あれば）→ 参考年齢・参考性別
    → 末尾の 1 行**。来歴（§10.8）は出せた状態でも出せない状態でも同じ位置に
    置く——「今どのモデルの話をしているか」は、数値が出ていない間こそ読めた
    ほうがよい。

    ``error`` は**ゲートより後ろで見る**。ゲートが隠している間に推論の失敗理由を
    代わりに見せると、ゲートが働いた回と推論が落ちた回を利用者が区別できてしまう
    （§10.11-2）。

    **引数だけを読み、pipeline も session state も触らない。** 状態ごとに 1 回の
    描画として検証できるようにするためである。
    """
    with st.container(border=True):
        st.caption(DEMOGRAPHIC_RUNTIME_CAPTION)
        if not available:
            st.info(MODEL_MISSING_NOTICE)
            return
        if not show_demographics:
            st.info(DISPLAY_OFF_NOTICE)
            return

        view = resolve_reference_age_view(
            estimate=estimate,
            gate_outcome=gate_outcome,
            show_demographics=True,
        )
        if view.estimate is not None:
            render_demographic_estimate(view.estimate)
        elif view.notice:
            st.info(view.notice)
        elif error:
            st.error(error)
        else:
            st.info(pending_notice)

        if footnote:
            st.caption(footnote)
