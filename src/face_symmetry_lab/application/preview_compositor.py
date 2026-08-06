"""表示専用の合成処理（§14.3b, §1.2(17)）。

`RealtimeLandmarkPipeline` から切り出した。ここに集めたのは「画面に出る絵を
作る」責務だけである。ランドマークと正中線の描画、背景の置換、そして解析を
飛ばしたフレームが直前の結果を再利用するためのキャッシュを持つ。

どれも測定値を変えない（FR-027, FR-028）。正中線と背景は解析の *後* に描かれ、
明るさ補正は検出器へ渡すフレームにしか掛からない。表示設定を 1 つの lock の
下にまとめることで、その主張を検証しやすい形に保つ。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.application.ports import SegmentationProvider
from face_symmetry_lab.domain.frame_adjustments import (
    BACKGROUND_COLORS,
    BackgroundMode,
    FrameAdjustmentError,
    apply_brightness_gain,
    composite_background,
)
from face_symmetry_lab.domain.landmarks import LandmarkFrame
from face_symmetry_lab.domain.overlay import (
    DEFAULT_MIDLINE_COLOR,
    DEFAULT_POINT_COLOR,
    DEFAULT_POINT_RADIUS,
    draw_face_midline,
    draw_landmark_points,
)
from face_symmetry_lab.runtime.display_cache import TtlCache


class PreviewCompositor:
    """描画・背景置換・表示キャッシュをまとめて受け持つ。

    表示 *設定*（明るさ・正中線・背景モード）を 1 つの lock が守る。下の 2 つの
    キャッシュはそれぞれ自前の lock を持つので、この lock が設定と、そこから
    導かれた画素の両方を兼ねることはない。
    """

    def __init__(
        self,
        *,
        segmenter: SegmentationProvider | None = None,
        clock: Callable[[], float],
        midline_indices: tuple[int, ...] = (),
        midline_enabled: bool = True,
        midline_color: tuple[int, int, int] = DEFAULT_MIDLINE_COLOR,
        point_color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
        point_radius: int = DEFAULT_POINT_RADIUS,
        brightness_gain: float = 1.0,
        background_mode: BackgroundMode = BackgroundMode.OFF,
    ) -> None:
        # §14.3b: 使い捨ての 1 画素へ一度適用して検証する。`apply_brightness_gain`
        # 自身が課す境界をそのまま再利用できる。
        apply_brightness_gain(np.zeros((1, 1, 3), dtype=np.uint8), brightness_gain)
        self._segmenter = segmenter
        self._clock = clock
        self._midline_indices = midline_indices
        self._midline_enabled = midline_enabled
        self._midline_color = midline_color
        self._point_color = point_color
        self._point_radius = point_radius
        self._brightness_gain = brightness_gain
        self._background_mode = background_mode
        self._lock = threading.Lock()
        #: §1.2(17): 間引かれたフレームが表示してよいもの。どちらも呼び出し側が
        #: 渡す同じ TTL で失効する。
        self._geometry_cache: TtlCache[LandmarkFrame] = TtlCache()
        self._mask_cache: TtlCache[NDArray[np.float32]] = TtlCache()

    # --- 表示設定 -------------------------------------------------------------

    def set_midline_enabled(self, enabled: bool) -> None:
        """表示専用の中心線の設定をスレッドセーフに更新する。"""
        with self._lock:
            self._midline_enabled = bool(enabled)

    def set_background_mode(self, mode: BackgroundMode) -> None:
        """表示専用の背景置換をスレッドセーフに更新する。"""
        with self._lock:
            self._background_mode = mode

    def set_brightness_gain(self, gain: float) -> None:
        """検出器の入力にだけ掛ける明るさ補正を更新する（§14.3c）。"""
        with self._lock:
            self._brightness_gain = gain

    @property
    def brightness_gain(self) -> float:
        """検出器の入力にだけ適用されている現在の補正値（§14.3b）。

        表示出力には決して適用されないため、これは利用者が見られるものではなく
        AI の「見えない検出補助」の読み戻しである。
        """
        with self._lock:
            return self._brightness_gain

    # --- 検出器への入力 -------------------------------------------------------

    def brighten_for_detection(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """ランドマーク検出器のためだけに明るくしたコピーを返す（§14.3b）。"""
        with self._lock:
            gain = self._brightness_gain
        if gain == 1.0:
            return rgb_image
        try:
            return apply_brightness_gain(rgb_image, gain)
        except FrameAdjustmentError:
            # gain は設定時に検証済みなので、ここでの失敗はフレーム自体が
            # 使えないことを意味する。元画像のまま検出を失敗させ、本当の
            # 問題を覆い隠さない。
            return rgb_image

    # --- 描画 -----------------------------------------------------------------

    def paint(self, rgb_image: NDArray[np.uint8], frame: LandmarkFrame) -> NDArray[np.uint8]:
        """ランドマークと、有効なら正中線を描く（FR-013）。

        描くものが無いときは `rgb_image` 自体を返し、呼び出し側が変更していない
        フレームの再エンコードを省けるようにする。
        """
        if not frame.points:
            return rgb_image
        painted = draw_landmark_points(
            rgb_image,
            frame,
            color=self._point_color,
            radius=self._point_radius,
        )
        with self._lock:
            midline_enabled = self._midline_enabled
        if midline_enabled:
            painted = draw_face_midline(
                painted,
                frame,
                midline_indices=self._midline_indices,
                color=self._midline_color,
            )
        return painted

    # --- 表示キャッシュ -------------------------------------------------------

    def remember_geometry(self, frame: LandmarkFrame, *, at: float) -> None:
        """単一顔フレームの幾何を保持し、それ以外ではキャッシュを捨てる。"""
        if frame.has_single_face:
            self._geometry_cache.store(frame, at=at)
        else:
            self._geometry_cache.clear()

    def clear_caches(self) -> None:
        """幾何と mask のキャッシュを両方捨てる。"""
        self._geometry_cache.clear()
        self._mask_cache.clear()

    def clear_geometry_cache(self) -> None:
        """幾何キャッシュだけを捨てる（フレーム単位の失敗時）。"""
        self._geometry_cache.clear()

    def draw_cached(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        ttl_seconds: float,
    ) -> NDArray[np.uint8]:
        """間引いたフレームを、TTL の内側で直前の解析結果から描き直す。"""
        painted = self._draw_cached_geometry(rgb_image, ttl_seconds=ttl_seconds)
        return self._composite_cached_background(painted, ttl_seconds=ttl_seconds)

    def _draw_cached_geometry(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        ttl_seconds: float,
    ) -> NDArray[np.uint8]:
        frame = self._geometry_cache.read(now=self._clock(), ttl_seconds=ttl_seconds)
        if frame is None:
            return rgb_image
        return self.paint(rgb_image, frame)

    def _composite_cached_background(
        self,
        painted: NDArray[np.uint8],
        *,
        ttl_seconds: float,
    ) -> NDArray[np.uint8]:
        """間引いたフレームで直前の mask を再利用する。使えなければ素通しする。

        背景置換が OFF のとき、mask が無いとき、mask が失効しているときは
        ``painted`` をそのまま返す。
        """
        with self._lock:
            mode = self._background_mode
        if mode is BackgroundMode.OFF:
            return painted
        mask = self._mask_cache.read(now=self._clock(), ttl_seconds=ttl_seconds)
        if mask is None:
            return painted
        try:
            return composite_background(painted, mask, BACKGROUND_COLORS[mode])
        except Exception:
            return painted

    # --- 背景置換 -------------------------------------------------------------

    def apply_background(
        self,
        source_image: NDArray[np.uint8],
        painted: NDArray[np.uint8],
        timestamp_ms: int,
        *,
        at: float,
    ) -> tuple[NDArray[np.uint8], float]:
        """表示フレームにだけ単色背景を合成する（§14.3b）。

        フレームと、segmentation に要した時間（§14.3f）を返す。置換が OFF か
        利用不可で何も走らなかった場合は ``0.0`` で、``painted`` はそのまま
        返る。

        segmentation は ``painted`` ではなく元フレームに対して走らせる。
        オーバーレイの描画が mask へ漏れないようにするためである。ここでの
        失敗は表示だけの問題であり、通信の失敗と取り違えてはならない
        （NFR-017）。握り潰し、合成前のフレームを返す。
        """
        with self._lock:
            mode = self._background_mode
            segmenter = self._segmenter
        if mode is BackgroundMode.OFF or segmenter is None:
            return painted, 0.0
        started = self._clock()
        try:
            mask = segmenter.segment(source_image, timestamp_ms=timestamp_ms)
            composited = composite_background(painted, mask, BACKGROUND_COLORS[mode])
        except Exception:
            return painted, (self._clock() - started) * 1000.0
        self._mask_cache.store(mask, at=at)
        return composited, (self._clock() - started) * 1000.0

    def close(self) -> None:
        """キャッシュを捨て、segmenter を解放する。"""
        self.clear_caches()
        if self._segmenter is not None:
            self._segmenter.close()
