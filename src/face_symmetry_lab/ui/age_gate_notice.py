"""FR-038 / AC-AGEGATE-04: 参考年齢・参考性別を「出すか、何と言うか」の 1 か所。

**このゲートで最も危険なのは数値ではなく文言である。** ゲートの内部判断
（`application.age_gate.AgeGateOutcome.reason`）を利用者へそのまま見せると、
「18歳未満と判定されました」のような人物の属性断定になり得る。ここが持つのは
「参考年齢を表示できません」という、アプリ側の都合を述べるだけの固定文言であり、
内部判断（下限側の発火・上限側の発火・判定不能）を区別しない。

**判定そのものもここが持つ**（2026-08-26、§5.1）。映像上の HUD・画像上の吹き出し・
Camera の専用タブ・FR-042 のレポートは、いずれも `resolve_reference_age_view` を
通る。区画ごとにゲートを読み直すと、片方だけが分岐を落としても「動いているように
見える」——そしてその区画からだけ、ゲートが隠した判断が読めてしまう（§10.12-3）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus
from face_symmetry_lab.domain.models import DemographicEstimate

#: `IN_RANGE`（表示してよい）と `NOT_EVALUATED`（呼んでいない）は何も言わない。
#: 通過を成功として祝わず、呼んでいないことについて言及もしない（§10.7）。
_NO_NOTICE: Final[str] = ""

#: `OUT_OF_RANGE` と `UNDETERMINED` は同一の文言にする（§10.11-2）。下限側の
#: 発火・上限側の発火・判定不能を利用者から見て区別させないためである。
#:
#: 主語を「この映像・画像」にしてあるのは、**同じ 1 文を Camera でも読めるように
#: するため**である（2026-08-26）。「この画像では」だけだと、ライブ映像の HUD に
#: 出たときに何を指しているのか分からない。範囲を今の入力へ限る働きは変わらない
#: ——人物についての言明にしないことがこの 1 文の役目である。
_HIDDEN_NOTICE: Final[str] = (
    "この映像・画像では参考年齢・参考性別を表示できません。左右差の分析結果には影響しません。"
)

#: §9.6.1: 1 セッションを通して 1 度も表示できなかったときに、FR-042 のレポートへ
#: 置く 1 行。per-frame の文言と同じく、できなかった内訳は述べない（§10.11-2）。
SESSION_HIDDEN_NOTICE: Final[str] = (
    "このセッションでは参考年齢・参考性別を表示できませんでした。左右差の分析結果には影響しません。"
)

AGE_GATE_NOTICE: Final[Mapping[AgeGateStatus, str]] = {
    AgeGateStatus.IN_RANGE: _NO_NOTICE,
    AgeGateStatus.OUT_OF_RANGE: _HIDDEN_NOTICE,
    AgeGateStatus.UNDETERMINED: _HIDDEN_NOTICE,
    AgeGateStatus.NOT_EVALUATED: _NO_NOTICE,
}


def age_gate_notice(status: AgeGateStatus) -> str:
    """状態に応じた固定文言を返す。未知の状態は安全側で空文字にする。"""
    return AGE_GATE_NOTICE.get(status, _NO_NOTICE)


@dataclass(frozen=True, slots=True)
class ReferenceAgeView:
    """ある瞬間に、参考年齢・参考性別として**画面へ出してよいもの**。

    `estimate` が `None` のときは数値を出さない。`notice` はその代わりに置く
    固定文言で、空文字なら何も言わない（利用者が表示を切っただけ、あるいは
    ゲートを呼んでいないだけの状態を「失敗」に見せないため、FR-024）。
    """

    estimate: DemographicEstimate | None
    notice: str

    @property
    def is_shown(self) -> bool:
        """FR-039 の参照平均を解決してよいか（§10.12-3 の判定と同じ）。"""
        return self.estimate is not None


def resolve_reference_age_view(
    *,
    estimate: DemographicEstimate | None,
    gate_outcome: AgeGateOutcome | None,
    show_demographics: bool = True,
) -> ReferenceAgeView:
    """参考年齢・参考性別の表示可否を、画面のどの区画でも同じ形で決める。

    順序は 3 つだけである。

    1. 利用者が表示を切っている（``show_demographics`` が偽）なら、数値も文言も
       出さない。**ゲートの文言をここへ出さない**——切っただけの状態を「表示
       できない」と説明すると、機能が失敗しているように読める（FR-024）。
    2. ゲートが表示を許さない（`may_show_reference_age` が偽）なら、推定が成功
       していても数値を捨て、`age_gate_notice` の固定文言を返す。`NOT_EVALUATED`
       はここで文言が空になる——呼んでいないことについては何も言わない。
    3. それ以外は推定をそのまま返す。``gate_outcome`` が `None`（ゲート未接続の
       pipeline）は従来どおり表示側へ倒す。
    """
    if not show_demographics:
        return ReferenceAgeView(estimate=None, notice=_NO_NOTICE)
    if gate_outcome is not None and not gate_outcome.may_show_reference_age:
        return ReferenceAgeView(estimate=None, notice=age_gate_notice(gate_outcome.status))
    return ReferenceAgeView(estimate=estimate, notice=_NO_NOTICE)
