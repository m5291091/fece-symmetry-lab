"""§14.3g の一発取り込み用バッファ。

`RealtimeLandmarkPipeline` から切り出した。「解析しなかったフレームが何を
描いてよいか」を答える表示キャッシュとは問いが異なり、こちらは「利用者が
どのフレームを残すよう求めたか」を答える。TTL で失効せず、消費されて消える。

要求が無い限り 1 枚もコピーしないことが、この機能を無償に保つ条件である。
常に保持する実装にすると、押されないかもしれないボタンのために解析フレーム
ごとの全解像度コピーを払うことになる。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.landmarks import LandmarkFrame


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """§14.3g の一発対称化のために保持した、解析済みフレーム 1 枚。

    保持するのは加工前の *生* 画像であり、描画済みの表示フレームではない。
    対称化のオーバーレイは自前の幾何を描くため、すでに注記の入ったフレームへ
    重ねると 2 世代分のランドマークが同時に見えてしまう。
    """

    rgb_image: NDArray[np.uint8]
    frame: LandmarkFrame


class CaptureBuffer:
    """取り込み要求のアーミングと、応じたフレーム 1 枚の保持を担う。

    自前の lock を持つ。realtime 経路（`store`）と UI 経路（`take` など）から
    触られるためである。
    """

    def __init__(self) -> None:
        self._requested = False
        self._captured: CapturedFrame | None = None
        self._lock = threading.Lock()

    def request(self) -> None:
        """次に解析へ成功した単一顔フレームを保持するよう要求する（§14.3g）。"""
        with self._lock:
            self._requested = True

    def store(self, rgb_image: NDArray[np.uint8], frame: LandmarkFrame) -> None:
        """このフレームで要求を満たせるなら応じる。

        画像をコピーするのは、呼び出し元が WebRTC のバッファを所有しており、
        次のフレームで再利用するためである。受け付けるのは単一顔フレームだけに
        する。対称化には正中線が 1 本必要であり、対称化できない取り込みは
        要求だけ消費して後で失敗するだけになる。
        """
        if not frame.has_single_face:
            return
        with self._lock:
            if not self._requested:
                return
            self._requested = False
            self._captured = CapturedFrame(rgb_image=rgb_image.copy(), frame=frame)

    def take(self) -> CapturedFrame | None:
        """保持したフレームを消費する。準備できていなければ None を返す。

        覗き見ではなく消費にすることで、生画像が UI に実際に使われるまでの
        間しかメモリへ残らないようにする（§10.4, FR-034）。
        """
        with self._lock:
            captured = self._captured
            self._captured = None
            return captured

    @property
    def pending(self) -> bool:
        """要求がまだ使えるフレームを待っている状態かどうか。"""
        with self._lock:
            return self._requested

    def clear(self) -> None:
        """保留中の要求と保持済みフレームを両方破棄する（§4.4, §10.4）。"""
        with self._lock:
            self._requested = False
            self._captured = None
