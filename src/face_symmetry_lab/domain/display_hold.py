"""FR-041: 表示へ出す値を、一定時間だけ差し替えないでおく（表示ラッチ）。

§8.7 の EMA が変える量に対して、こちらが変えるのは **頻度** である。EMA は
連続時間のフィルタなので、`tau` をどれだけ伸ばしても平滑値は新しいサンプルの
たびに動く。読んでいる最中に数字が止まっていてほしいという要求は、平滑化では
満たせない（ADR-0008）。

**この module は測定を一切知らない。** 保持するかどうかは「時刻・保持時間・鍵」
だけで決まる純粋計算であり、時計は呼び出し側が渡す。

鍵（``key``）には**「値が届いているかどうかの型」**を入れる約束である。品質
ゲートの状態、各測定値の有無——これが変われば保持を打ち切り、その場で新しい値を
返す。この約束が、顔を見失った後も古い数値が画面に残る事故を防いでいる
（不変条件 11、AC-HOLD-02）。保持してよいのは「同じものが測れ続けている間の、
値の細かい揺れ」だけである。
"""

from __future__ import annotations

from math import isfinite

#: FR-041 の離散段。0 は「保持しない」で、既定でもある——この機能を知らない
#: 利用者の画面が変わらないようにするためである。
DISPLAY_HOLD_CHOICES_SECONDS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
DEFAULT_DISPLAY_HOLD_SECONDS: float = 0.0
MAX_DISPLAY_HOLD_SECONDS: float = 5.0


class DisplayHoldError(ValueError):
    """表示ラッチの設定が不正である。"""


def validate_hold_seconds(value: float) -> float:
    """FR-041 の設定値を検証して返す。**clamp しない。**

    `domain.smoothing.validate_tau_seconds` と同じ理由である——範囲外を境界へ
    丸めると、利用者の指定を黙って別の設定へ読み替えることになる。
    """
    if not isfinite(value):
        raise DisplayHoldError("表示の更新間隔は有限値である必要があります。")
    if not 0.0 <= value <= MAX_DISPLAY_HOLD_SECONDS:
        raise DisplayHoldError(
            f"表示の更新間隔は 0〜{MAX_DISPLAY_HOLD_SECONDS} 秒の範囲である必要があります。"
        )
    return float(value)


class DisplayHold[T]:
    """直前に publish した値を、保持時間のあいだ返し続ける。

    保持の単位は呼び出し側が決める。`ui`/`application` はここへ **1 フレーム分の
    解析結果そのもの**を渡す（ADR-0008 (a)）。数値ごとに別々のラッチを持つと、
    同じ画面の別々の場所が別々の瞬間の測定を出す状態が生まれ、§8.6.3 の計算過程
    と部位別表示の文字列一致（AC-DISPLAY-04）が保持区間の境界でだけ壊れる。
    """

    def __init__(self, *, hold_seconds: float = DEFAULT_DISPLAY_HOLD_SECONDS) -> None:
        self._hold_seconds = validate_hold_seconds(hold_seconds)
        self._value: T | None = None
        self._key: object = None
        self._at: float | None = None
        self._primed = False

    @property
    def hold_seconds(self) -> float:
        return self._hold_seconds

    def set_hold_seconds(self, hold_seconds: float) -> None:
        """保持時間を差し替える。次の `update` から効く。

        保持中の値は捨てない。捨てると、設定を選び直すたびに表示が 1 度飛ぶ。
        新しい保持時間は、次に publish した時刻から数え直される。
        """
        self._hold_seconds = validate_hold_seconds(hold_seconds)

    def reset(self) -> None:
        """保持中の値を捨てる（§4.4 のセッションリセットと同じ扱い）。"""
        self._value = None
        self._key = None
        self._at = None
        self._primed = False

    def update(self, value: T, *, at: float, key: object) -> T:
        """新しい値を渡し、いま画面へ出すべき値を返す。

        次のいずれかなら、渡された値をそのまま publish する。

        * 保持時間が 0（既定）
        * まだ 1 度も publish していない
        * **鍵が変わった**（欠測への遷移・欠測からの復帰。AC-HOLD-02）
        * 時計が巻き戻った（表示が固まったままにならないように）
        * 前回の publish から保持時間が経過した

        それ以外では直前に publish した値を返し、内部時計も動かさない。
        呼ばれるたびに時計を進める実装だと、保持がずるずる伸びて更新されない。
        """
        if not isfinite(at):
            raise DisplayHoldError("表示ラッチの時刻は有限値である必要があります。")
        if (
            self._hold_seconds <= 0.0
            or not self._primed
            or self._at is None
            or key != self._key
            or at < self._at
            or at - self._at >= self._hold_seconds
        ):
            self._value = value
            self._key = key
            self._at = at
            self._primed = True
            return value
        # `T` が `None` を取りうるため、`self._value` の型は `T | None` になって
        # いる。ここへ来る経路では必ず publish 済みなので、その値をそのまま返す。
        return self._value  # type: ignore[return-value]
