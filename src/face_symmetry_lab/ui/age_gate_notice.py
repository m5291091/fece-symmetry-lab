"""FR-038 / AC-AGEGATE-04: 年齢ゲート発火時の固定表示文言（§10.11-2）。

**このゲートで最も危険なのは数値ではなく文言である。** ゲートの内部判断
（`application.age_gate.AgeGateOutcome.reason`）を利用者へそのまま見せると、
「18歳未満と判定されました」のような人物の属性断定になり得る。ここが持つのは
「この画像では参考年齢を表示できません」という、アプリ側の都合を述べるだけの
固定文言であり、内部判断（下限側の発火・上限側の発火・判定不能）を区別しない。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from face_symmetry_lab.application.age_gate import AgeGateStatus

#: `IN_RANGE`（表示してよい）と `NOT_EVALUATED`（呼んでいない）は何も言わない。
#: 通過を成功として祝わず、呼んでいないことについて言及もしない（§10.7）。
_NO_NOTICE: Final[str] = ""

#: `OUT_OF_RANGE` と `UNDETERMINED` は同一の文言にする（§10.11-2）。下限側の
#: 発火・上限側の発火・判定不能を利用者から見て区別させないためである。
_HIDDEN_NOTICE: Final[str] = (
    "この画像では参考年齢・参考性別を表示できません。左右差の分析結果には影響しません。"
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
