from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from face_symmetry_lab.adapters.inference.tracking_zoom_landmarker import (
    TrackingZoomLandmarker,
)
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.tracking_zoom import TrackingZoomPolicy

#: fake が受け取る *どんな* 画像でも、この比の位置に顔があるものとする。幾何の
#: 検証を、実際の検出内容から独立させるためである。
_FACE_FRACTION = ((0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6))


class _FakeLandmarkProvider:
    """画素の内容を無視し、呼び出しごとに用意した結果を返す。"""

    def __init__(self, modes: list[str]) -> None:
        self._modes = list(modes)
        self.received_shapes: list[tuple[int, int]] = []
        self.closed = False

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        height, width = rgb_image.shape[:2]
        self.received_shapes.append((height, width))
        mode = self._modes.pop(0) if self._modes else "single"
        if mode == "none":
            return build_landmark_frame(frame_id=frame_id, face_count=0, width=width, height=height)
        if mode == "multi":
            return build_landmark_frame(frame_id=frame_id, face_count=2, width=width, height=height)
        return build_landmark_frame(
            frame_id=frame_id,
            face_count=1,
            points=_FACE_FRACTION,
            width=width,
            height=height,
        )

    def close(self) -> None:
        self.closed = True


def _blank(height: int = 800, width: int = 1000) -> NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_the_first_frame_is_detected_on_the_full_frame_unscaled() -> None:
    fake = _FakeLandmarkProvider(["single"])
    zoomer = TrackingZoomLandmarker(fake)

    result = zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)

    assert fake.received_shapes == [(800, 1000)]
    assert result.points == _FACE_FRACTION
    assert result.width == 1000
    assert result.height == 800


def test_the_second_call_detects_on_a_cropped_and_enlarged_window() -> None:
    fake = _FakeLandmarkProvider(["single", "single"])
    policy = TrackingZoomPolicy(
        margin_ratio=0.6, min_window_size=0.18, target_px=480, max_scale=4.0
    )
    zoomer = TrackingZoomLandmarker(fake, policy=policy)

    zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)
    second = zoomer.detect_video(_blank(800, 1000), frame_id=2, timestamp_ms=33)

    # bbox (0.4,0.4)-(0.6,0.6) を余白 0.6 で広げると窓は (0.34,0.34)-(0.66,0.66)。
    # 1000x800 のフレーム上で 320x256px の切り出しとなり、1.5 倍に拡大される
    # （最長辺 320 -> 480）。
    assert fake.received_shapes[1] == (384, 480)
    # 窓(0.34..0.66) + 0.4*幅(0.32) == 0.468、+ 0.6*0.32 == 0.532。
    expected = ((0.468, 0.468), (0.532, 0.468), (0.468, 0.532), (0.532, 0.532))
    assert np.asarray(second.points) == pytest.approx(np.asarray(expected))
    assert second.width == 1000
    assert second.height == 800


def test_losing_the_face_falls_back_to_the_full_frame_next_time() -> None:
    fake = _FakeLandmarkProvider(["single", "none", "single"])
    zoomer = TrackingZoomLandmarker(fake)

    zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)
    lost = zoomer.detect_video(_blank(800, 1000), frame_id=2, timestamp_ms=33)
    recovered = zoomer.detect_video(_blank(800, 1000), frame_id=3, timestamp_ms=66)

    assert lost.face_count == 0
    assert lost.points == ()
    # 復帰時は古い切り出しではなく、フレーム全体を改めて探索しなければならない。
    assert fake.received_shapes[2] == (800, 1000)
    assert recovered.points == _FACE_FRACTION


def test_a_periodic_recheck_returns_to_the_full_frame_even_without_losing_the_face() -> None:
    fake = _FakeLandmarkProvider(["single", "single", "single"])
    policy = TrackingZoomPolicy(recheck_interval_ms=100)
    zoomer = TrackingZoomLandmarker(fake, policy=policy)

    zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)
    zoomer.detect_video(_blank(800, 1000), frame_id=2, timestamp_ms=50)
    zoomer.detect_video(_blank(800, 1000), frame_id=3, timestamp_ms=150)

    assert fake.received_shapes[0] == (800, 1000)  # 初回
    assert fake.received_shapes[1] != (800, 1000)  # ズーム中で、まだ再確認の時刻ではない
    assert fake.received_shapes[2] == (800, 1000)  # t=150 で再確認を強制


def test_a_second_face_found_on_a_recheck_is_reported_and_tracking_is_dropped() -> None:
    fake = _FakeLandmarkProvider(["single", "single", "multi", "single"])
    policy = TrackingZoomPolicy(recheck_interval_ms=100)
    zoomer = TrackingZoomLandmarker(fake, policy=policy)

    zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)
    zoomer.detect_video(_blank(800, 1000), frame_id=2, timestamp_ms=50)
    multi = zoomer.detect_video(_blank(800, 1000), frame_id=3, timestamp_ms=150)
    after = zoomer.detect_video(_blank(800, 1000), frame_id=4, timestamp_ms=200)

    assert fake.received_shapes[2] == (800, 1000)  # 再確認は場面全体を見た
    assert multi.face_count == 2
    assert multi.points == ()
    # 複数顔を検出した後、直前の切り出しで黙って追跡を再開してはならない。
    assert fake.received_shapes[3] == (800, 1000)
    assert after.points == _FACE_FRACTION


def test_update_policy_applies_to_the_existing_tracked_face_immediately() -> None:
    """方針の変更（Advanced 設定など）が追跡を破棄してはならない。

    直前に判明した顔の位置はなお有効なので、全フレーム再探索を強制せず、
    新しい余白と窓を直ちにその位置へ適用すべきである。
    """
    fake = _FakeLandmarkProvider(["single", "single"])
    zoomer = TrackingZoomLandmarker(fake, policy=TrackingZoomPolicy(margin_ratio=0.6))

    zoomer.detect_video(_blank(800, 1000), frame_id=1, timestamp_ms=0)
    # target_px=1 にすることで、この検証を（別途テストする）拡大処理から切り離す。
    zoomer.update_policy(TrackingZoomPolicy(margin_ratio=0.0, min_window_size=0.2, target_px=1))
    zoomer.detect_video(_blank(800, 1000), frame_id=2, timestamp_ms=33)

    # bbox (0.4,0.4)-(0.6,0.6) は 0.2x0.2。余白 0.0 なので 0.2 のままで、
    # min_window_size 0.2 の下限にも一致するため、切り出しは生の bbox と等しい。
    assert fake.received_shapes[1] == (round(0.2 * 800), round(0.2 * 1000))


def test_policy_reads_back_the_constructor_default() -> None:
    zoomer = TrackingZoomLandmarker(_FakeLandmarkProvider([]))

    assert zoomer.policy == TrackingZoomPolicy()


def test_policy_reads_back_the_latest_update_policy_call() -> None:
    zoomer = TrackingZoomLandmarker(_FakeLandmarkProvider([]))
    updated = TrackingZoomPolicy(margin_ratio=0.1, target_px=600)

    zoomer.update_policy(updated)

    assert zoomer.policy == updated


def test_close_closes_the_wrapped_provider() -> None:
    fake = _FakeLandmarkProvider(["single"])
    zoomer = TrackingZoomLandmarker(fake)

    zoomer.close()

    assert fake.closed is True
