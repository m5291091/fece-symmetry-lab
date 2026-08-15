"""FR-038 / AC-AGEGATE-04: ゲート発火時の表示文言。

**このゲートで最も危険なのは数値ではなく文言である。** 内部判断をそのまま日本語に
すると「18歳未満と判定されました」になり、それは人物の属性を断定する表示であって
`CLAUDE.md` の禁止事項（本人確認・人物評価）に触れる。出してよいのは「参考年齢を
表示できない」という**アプリ側の都合**までである。
"""

from __future__ import annotations

import pytest

from face_symmetry_lab.application.age_gate import AgeGateStatus
from face_symmetry_lab.ui.age_gate_notice import (
    AGE_GATE_NOTICE,
    age_gate_notice,
)

#: 人物の属性を断定する語。1 つでも表示文言に現れてはならない（§10.11-2）。
FORBIDDEN = (
    "未成年",
    "18歳未満",
    "18 歳未満",
    "子供",
    "子ども",
    "児童",
    "幼児",
    "高齢者",
    "お年寄り",
    "歳に見え",
    "才に見え",
    "あなたは",
    "年齢認証",
    "本人確認",
)

#: 価値判断語（§10.7）。
JUDGEMENT = ("美しい", "良い", "悪い", "優れ", "理想", "劣")


def test_every_status_has_an_entry_in_the_notice_table() -> None:
    """状態を足したときに文言を書き忘れると、UI が黙って何も出さなくなる。

    **「エントリがあること」と「文言が空でないこと」は別の要求である。**
    `IN_RANGE` と `NOT_EVALUATED` は意図的に空文字（何も言わない）なので、
    ここで見るのは表に漏れが無いことと、例外を投げず `str` を返すことだけ。
    「発言すべき状態が実際に発言すること」は
    `test_the_gate_states_that_speak_actually_say_something` が見る。
    """
    for status in AgeGateStatus:
        assert status in AGE_GATE_NOTICE, status
        assert isinstance(age_gate_notice(status), str), status


def test_the_gate_states_that_speak_actually_say_something() -> None:
    """参考年齢を出せない 2 状態では、必ず理由を表示する（無言で消さない）。"""
    for status in (AgeGateStatus.OUT_OF_RANGE, AgeGateStatus.UNDETERMINED):
        assert age_gate_notice(status).strip(), status


@pytest.mark.parametrize("status", list(AgeGateStatus))
def test_no_notice_asserts_an_attribute_about_the_person(status: AgeGateStatus) -> None:
    """AC-AGEGATE-04: 属性を断定する語を 1 つも含まない。"""
    text = age_gate_notice(status)

    for word in FORBIDDEN:
        assert word not in text, f"{status.name} の文言に属性断定が含まれる: {word}"


@pytest.mark.parametrize("status", list(AgeGateStatus))
def test_no_notice_uses_value_judgement_words(status: AgeGateStatus) -> None:
    text = age_gate_notice(status)

    for word in JUDGEMENT:
        assert word not in text, f"{status.name} の文言に価値判断語が含まれる: {word}"


def test_out_of_range_and_undetermined_are_indistinguishable_to_the_reader() -> None:
    """下限側の発火・上限側の発火・判定不能を区別して見せない（§10.11-2）。

    罠: 「範囲外」と「判定できません」を別文言にすると、利用者は前者を
    「自分は 18〜75 歳の外だと言われた」と読む。**アプリはどちらの場合も
    「参考年齢を表示できない」としか言わない。**
    """
    assert age_gate_notice(AgeGateStatus.OUT_OF_RANGE) == age_gate_notice(
        AgeGateStatus.UNDETERMINED
    )


def test_the_notice_explains_that_only_the_reference_age_is_affected() -> None:
    """左右差の分析は続いていることを利用者へ伝える（不変条件 12 の可視化）。"""
    text = age_gate_notice(AgeGateStatus.OUT_OF_RANGE)

    assert "参考年齢" in text
    assert "左右差" in text


def test_in_range_shows_no_notice() -> None:
    """通ったときは何も言わない。通過を成功として祝わない（§10.7）。"""
    assert age_gate_notice(AgeGateStatus.IN_RANGE) == ""
    assert age_gate_notice(AgeGateStatus.NOT_EVALUATED) == ""
