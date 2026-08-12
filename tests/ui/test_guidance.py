import pytest

from face_symmetry_lab.domain.models import ReasonCode
from face_symmetry_lab.ui.guidance import REASON_GUIDANCE


@pytest.mark.parametrize("reason", list(ReasonCode))
def test_every_reason_code_has_a_japanese_guidance_message(reason: ReasonCode) -> None:
    """AC-GUIDE-01: `ReasonCode` の全メンバーに指示文が定義されている。

    `guidance_message`（ライブ解析パネル専用の 1 件取得ヘルパー）は項目26（H3）
    でパネルごと削除したため、唯一の呼び出し元になった詳細結果
    （`ui.scores.render_guidance`）と同じ辞書を直接検証する。
    """
    message = REASON_GUIDANCE.get(reason)

    assert message is not None
    assert message != ""


def test_guidance_is_actionable_for_common_reasons() -> None:
    """§6.3: 近すぎる・遠すぎるは逆方向の操作を指すので、文言も異ならなければ
    ならない。"""
    assert REASON_GUIDANCE[ReasonCode.TOO_FAR] != REASON_GUIDANCE[ReasonCode.TOO_CLOSE]
