"""年齢・性別 worker との受け渡しを担う中継役（§14.3, FR-022〜024）。

`RealtimeLandmarkPipeline` から切り出した。年齢帯の推論は左右差の解析とは
独立した周期で走り（§14.3）、失敗しても左右差分析を止めてはならない
（不変条件 12）。その「独立性」を 1 箇所に閉じ込めることで、realtime 経路
から年齢帯まわりの状態が消える。

自前の lock を持つ。表示可否のトグルは UI スレッドから、投入と回収は WebRTC
の受信スレッドから触られるためである。
"""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.image_input import ImageInputError, extract_face_crop
from face_symmetry_lab.application.ports import DemographicSampler
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.models import DemographicEstimate


class DemographicRelay:
    """worker への投入と、最新結果の再表示判定を担う。

    worker が ``None`` のとき（モデル未配置など）も同じ形で扱えるようにして
    あるので、呼び出し側に分岐が増えない。
    """

    def __init__(self, worker: DemographicSampler | None) -> None:
        self._worker = worker
        self._enabled = True
        #: (version, その版を最初に表示したときの elapsed_ms)。結果が出る前は
        #: None。1 つのフィールドにまとめることで両者が食い違えないようにする。
        self._seen: tuple[int, int] | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """このセッションで年齢・性別推論が使えるかどうか。

        pipeline 構築時に決まるので、UI は再度モデルを解決して食い違うことなく、
        実在する worker を条件にトグルを制御できる。
        """
        return self._worker is not None

    def set_enabled(self, enabled: bool) -> None:
        """年齢・性別推論の表示可否を切り替える（§15.1、既定は ON）。"""
        with self._lock:
            self._enabled = bool(enabled)

    def _active_worker(self) -> DemographicSampler | None:
        """利用者が表示を ON にしている間だけ worker を返す。"""
        if self._worker is None:
            return None
        with self._lock:
            return self._worker if self._enabled else None

    def submit(self, rgb_image: NDArray[np.uint8], frame: LandmarkFrame) -> None:
        """単一顔の切り出しを、クリティカルパス外の worker へ渡す（§14.3）。

        年齢・性別推論がこの WebRTC コールバックスレッドを塞ぐことはない。
        worker の周期は解析周期よりはるかに遅いので、先に worker へ問い合わせる。
        そうしないと大半のフレームが、捨てられる切り出しの費用を払うことになる。
        """
        worker = self._active_worker()
        if worker is None or not frame.has_single_face or not worker.is_due():
            return
        try:
            crop = extract_face_crop(rgb_image, frame)
        except ImageInputError:
            return
        worker.submit(crop)

    def collect(self, timestamp_ms: int) -> tuple[DemographicEstimate, int] | None:
        """worker の最新結果を読み、フレームをまたぐ再表示に印を付ける。

        `demographic_is_cached`（FR-023）は、このフレームでの新規推論と、同じ
        区間の直前結果の再表示とを区別する。worker は解析フレームよりずっと
        少ない頻度でしか走らないためである。
        """
        worker = self._active_worker()
        if worker is None:
            return None
        latest = worker.latest()
        if latest is None:
            return None
        estimate, version = latest
        with self._lock:
            seen = self._seen
            if seen is None or seen[0] != version:
                self._seen = (version, timestamp_ms)
                return estimate, timestamp_ms
            observed_elapsed_ms = seen[1]
        return estimate.model_copy(update={"is_cached": True}), observed_elapsed_ms

    def set_interval_ms(self, interval_ms: int) -> None:
        """worker 自身の推論周期を retune する（§15.2 `age_interval_ms`）。"""
        if self._worker is not None:
            self._worker.set_interval_ms(interval_ms)

    def reset(self) -> None:
        """比較区間が変わったので、キャッシュ済みの推定を破棄する（§8.8, FR-023）。"""
        if self._worker is not None:
            self._worker.reset()
        with self._lock:
            self._seen = None

    def close(self) -> None:
        """worker を解放する。"""
        if self._worker is not None:
            self._worker.close()
