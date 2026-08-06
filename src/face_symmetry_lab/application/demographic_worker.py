"""クリティカルパス外の年齢・性別推論（§14.3, FR-022〜024）。

年齢・性別の推論は専用のバックグラウンドスレッドで動き、``interval_ms`` に
よってランドマーク解析の周期から切り離される。同時に走る推論は最大 1 件
（容量 1）である。早すぎる投入や処理中の投入はキューへ積まず捨てるので、
フレームのコールバックが塞がることも、古い切り出しが溜まることもない。
（`runtime.frame_scheduler.FrameScheduler` はランドマーク解析に同じ規則を課すが、
その context manager API はこの受け渡しがまたぐ 2 つのスレッドをまたげない。）
完了前にリセットされた区間のために計算された結果は、完了時に破棄する。後続の
比較区間へ漏れ出さないようにするためである。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.ports import DemographicEstimator
from face_symmetry_lab.domain.models import DemographicEstimate

#: §15.2 の `age_interval_ms` の既定値。このマイルストーンでは利用者が編集できない。
DEFAULT_DEMOGRAPHIC_INTERVAL_MS = 1_000


class DemographicWorker:
    """容量 1 の背景年齢・性別推定器。自身の周期で投入を制御する。"""

    def __init__(
        self,
        *,
        estimator: DemographicEstimator,
        interval_ms: int = DEFAULT_DEMOGRAPHIC_INTERVAL_MS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_ms < 0:
            raise ValueError("interval_ms は 0 以上である必要があります。")
        self._estimator = estimator
        self._interval_seconds = interval_ms / 1000.0
        self._clock = clock
        self._state_lock = threading.Lock()
        self._next_submit_at = clock()
        self._generation = 0
        self._busy = False
        self._closed = False
        self._latest: DemographicEstimate | None = None
        self._latest_version = 0

    def is_due(self) -> bool:
        """いま ``submit`` したら実際に処理が始まるかどうかを返す。

        捨てられるだけの切り出しを作る手間を、呼び出し側が省けるようにする。
        フレームの周期は ``interval_ms`` よりはるかに速く、投入の大半は拒否
        されるためである。これは助言的な判定であり、``submit`` は lock の内側で
        改めて確認する。
        """
        with self._state_lock:
            return not self._closed and not self._busy and self._clock() >= self._next_submit_at

    def submit(self, rgb_crop: NDArray[np.uint8]) -> threading.Thread | None:
        """待機中かつ実行時刻に達していれば背景推論を開始する。そうでなければ何もしない。"""
        with self._state_lock:
            if self._closed or self._busy or self._clock() < self._next_submit_at:
                return None
            # 実行時刻の判定と同じ lock の内側で確保するので、2 つの呼び出しが
            # 同時に通過することはない。解放は `_run` の `finally` が行う。
            self._busy = True
            self._next_submit_at = self._clock() + self._interval_seconds
            generation = self._generation
        # 呼び出し側のバッファから切り離す。この背景スレッドが読む前に、pipeline
        # が元フレームを書き換えたり使い回したりする可能性があるためである。
        crop = np.array(rgb_crop, copy=True)
        thread = threading.Thread(target=self._run, args=(crop, generation), daemon=True)
        try:
            thread.start()
        except Exception:
            self._release()
            raise
        return thread

    def _run(self, crop: NDArray[np.uint8], generation: int) -> None:
        try:
            estimate: DemographicEstimate | None = self._estimator.estimate(crop)
        except Exception:
            # FR-024: 推論の失敗が左右差の解析へ影響してはならない。直前の正常な
            # 結果（あれば）がそのまま再利用され続けるだけである。
            estimate = None
        finally:
            self._release()
        if estimate is None:
            return
        with self._state_lock:
            if self._closed or generation != self._generation:
                return
            self._latest_version += 1
            self._latest = estimate

    def _release(self) -> None:
        with self._state_lock:
            self._busy = False

    def latest(self) -> tuple[DemographicEstimate, int] | None:
        """現時点で最良の推定と版トークンを返す。無ければ None。"""
        with self._state_lock:
            if self._latest is None:
                return None
            return self._latest, self._latest_version

    def set_interval_ms(self, interval_ms: int) -> None:
        """標本化の周期を変更する（§15.2 `age_interval_ms`）。

        即座に反映される。次の投入は直ちに許可され、それ以降の投入は新しい間隔に
        従う。実行中の推論には影響しない。
        """
        if interval_ms < 0:
            raise ValueError("interval_ms は 0 以上である必要があります。")
        with self._state_lock:
            self._interval_seconds = interval_ms / 1000.0
            self._next_submit_at = self._clock()

    def reset(self) -> None:
        """新しい区間を開始する。キャッシュ済みの結果と実行中のものを破棄する。"""
        with self._state_lock:
            self._generation += 1
            self._latest = None
            self._latest_version = 0
            self._next_submit_at = self._clock()

    def close(self) -> None:
        """投入の受け付けを止め、キャッシュ済みの結果を破棄する。"""
        with self._state_lock:
            self._closed = True
            self._generation += 1
            self._latest = None
