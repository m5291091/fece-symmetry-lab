"""FR-041: 表示の更新間隔（表示ラッチ、`domain.display_hold`）。

この部品が守るのは 2 点で、2 つ目のほうが重要である。

1. 保持時間の内側では、同じ値を返し続ける（画面の数値が動かない）。
2. **鍵が変わったら保持を打ち切る。** 鍵には「値が届いているかどうかの型」を
   入れる約束なので、顔を見失った・品質ゲートが閉じた瞬間に古い数値が画面へ
   残ることがない（不変条件 11、AC-HOLD-02）。ここが壊れると「測定が続いて
   いるという嘘」を表示する。
"""

from __future__ import annotations

import math

import pytest

from face_symmetry_lab.domain.display_hold import (
    DEFAULT_DISPLAY_HOLD_SECONDS,
    DISPLAY_HOLD_CHOICES_SECONDS,
    MAX_DISPLAY_HOLD_SECONDS,
    DisplayHold,
    DisplayHoldError,
    validate_hold_seconds,
)


def test_the_default_is_no_hold_at_all() -> None:
    """既定でこの機能を知らない利用者の画面が変わってはならない。"""
    assert DEFAULT_DISPLAY_HOLD_SECONDS == 0.0
    assert DEFAULT_DISPLAY_HOLD_SECONDS in DISPLAY_HOLD_CHOICES_SECONDS


def test_the_selectable_steps_are_sorted_and_inside_the_range() -> None:
    assert DISPLAY_HOLD_CHOICES_SECONDS == (0.0, 0.5, 1.0, 2.0, 5.0)
    assert list(DISPLAY_HOLD_CHOICES_SECONDS) == sorted(DISPLAY_HOLD_CHOICES_SECONDS)
    assert max(DISPLAY_HOLD_CHOICES_SECONDS) == MAX_DISPLAY_HOLD_SECONDS


@pytest.mark.parametrize("bad", [-0.5, 5.5, math.nan, math.inf])
def test_an_out_of_range_hold_is_refused_rather_than_clamped(bad: float) -> None:
    with pytest.raises(DisplayHoldError):
        validate_hold_seconds(bad)


def test_zero_seconds_passes_every_value_straight_through() -> None:
    """既定の挙動は「毎回更新」——従来と 1 ミリ秒も変わらない。"""
    hold = DisplayHold(hold_seconds=0.0)

    assert hold.update("a", at=0.0, key=1) == "a"
    assert hold.update("b", at=0.001, key=1) == "b"
    assert hold.update("c", at=0.002, key=1) == "c"


def test_a_value_is_kept_until_the_hold_elapses() -> None:
    hold = DisplayHold(hold_seconds=2.0)
    hold.update("first", at=10.0, key=1)

    assert hold.update("second", at=11.0, key=1) == "first"
    assert hold.update("third", at=11.9, key=1) == "first"


def test_the_next_value_is_published_once_the_hold_elapses() -> None:
    hold = DisplayHold(hold_seconds=2.0)
    hold.update("first", at=10.0, key=1)
    hold.update("ignored", at=11.0, key=1)

    assert hold.update("fresh", at=12.0, key=1) == "fresh"


def test_the_clock_restarts_from_the_value_actually_published() -> None:
    """罠: 呼ばれるたびに時計を進める実装は、保持がずるずる伸びて更新されない。"""
    hold = DisplayHold(hold_seconds=2.0)
    hold.update("first", at=10.0, key=1)
    hold.update("ignored", at=11.5, key=1)
    published = hold.update("second", at=12.0, key=1)
    assert published == "second"

    # 2 本目を出した 12.0 が新しい起点。13.9 はまだ保持の内側である。
    assert hold.update("third", at=13.9, key=1) == "second"
    assert hold.update("fourth", at=14.0, key=1) == "fourth"


def test_a_changed_key_breaks_the_hold_immediately() -> None:
    """AC-HOLD-02: 顔を見失った瞬間に古い数値が残ってはならない。

    鍵には「値が届いているかどうかの型」を入れる約束なので、欠測への遷移は
    ここで必ず鍵の変化として現れる。
    """
    hold = DisplayHold(hold_seconds=5.0)
    hold.update("measuring", at=10.0, key=("OK", True))

    assert hold.update("missing", at=10.2, key=("NO_FACE", False)) == "missing"


def test_a_recovery_is_published_immediately_too() -> None:
    """罠: 欠測への遷移だけを特別扱いする実装は、復帰を保持時間ぶん遅らせる。"""
    hold = DisplayHold(hold_seconds=5.0)
    hold.update("missing", at=10.0, key=("NO_FACE", False))

    assert hold.update("measuring", at=10.2, key=("OK", True)) == "measuring"


def test_the_hold_restarts_from_the_moment_the_key_changed() -> None:
    hold = DisplayHold(hold_seconds=2.0)
    hold.update("measuring", at=10.0, key=("OK", True))
    hold.update("missing", at=10.5, key=("NO_FACE", False))

    assert hold.update("still missing", at=11.0, key=("NO_FACE", False)) == "missing"
    assert hold.update("later", at=12.5, key=("NO_FACE", False)) == "later"


def test_time_going_backwards_publishes_rather_than_freezing() -> None:
    """単調時計が巻き戻る状況（テスト・再接続）で、表示が固まったままにならない。"""
    hold = DisplayHold(hold_seconds=5.0)
    hold.update("first", at=10.0, key=1)

    assert hold.update("second", at=9.0, key=1) == "second"


def test_reset_drops_whatever_was_being_held() -> None:
    """§4.4: セッションのリセットは、表示中の値も一緒に捨てる。"""
    hold = DisplayHold(hold_seconds=5.0)
    hold.update("first", at=10.0, key=1)

    hold.reset()

    assert hold.update("second", at=10.1, key=1) == "second"


def test_the_hold_length_can_be_changed_and_takes_effect_at_once() -> None:
    hold = DisplayHold(hold_seconds=5.0)
    hold.update("first", at=10.0, key=1)

    hold.set_hold_seconds(0.0)

    assert hold.hold_seconds == pytest.approx(0.0)
    assert hold.update("second", at=10.1, key=1) == "second"


def test_changing_the_hold_length_refuses_an_out_of_range_value() -> None:
    hold = DisplayHold(hold_seconds=2.0)

    with pytest.raises(DisplayHoldError):
        hold.set_hold_seconds(9.0)

    assert hold.hold_seconds == pytest.approx(2.0)


def test_none_is_a_value_like_any_other() -> None:
    """`None`（解析なし）も保持の対象になりうる。鍵が同じなら差し替えない。"""
    hold = DisplayHold(hold_seconds=2.0)

    assert hold.update(None, at=10.0, key=("NO_FACE", False)) is None
    assert hold.update(None, at=10.5, key=("NO_FACE", False)) is None
