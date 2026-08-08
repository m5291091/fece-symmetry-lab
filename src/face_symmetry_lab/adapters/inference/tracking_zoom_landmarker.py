"""映像用 `LandmarkProvider` を、追跡ズームの切り出し・拡大で包む（§14.3a）。

PC の webcam はスマートフォンの自撮りよりはるかに小さく顔を写す。この adapter
は直前に検出した顔の周りを切り出して拡大し、生フレームの代わりに包んだ provider
へ渡す。これにより、利用者がどれだけ離れて座っていても検出器は顔に対してより
高い実効解像度を得る。包んだ provider の動作自体は変えない。変えるのはそれが
見る画像だけであり、結果は全フレーム座標へ射影し直すので、下流の品質ゲート・
採点・オーバーレイは影響を受けない。

顔をとらえている間も、定期的に全フレーム検出を強制する（`recheck_interval_ms`）。
そうしないと現在の切り出しの外にいる顔が永久に見えないままになるためである。
2 人目の顔を黙って見えなくすることは、単一顔の方針（FR-008）に反する。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from face_symmetry_lab.application.ports import LandmarkProvider
from face_symmetry_lab.domain.landmarks import LandmarkFrame, build_landmark_frame
from face_symmetry_lab.domain.tracking_zoom import (
    FULL_FRAME,
    BoundingBox,
    TrackingZoomPolicy,
    bounding_box_of,
    next_zoom_window,
    remap_depths,
    remap_points,
    resize_scale_for,
)


class TrackingZoomLandmarker:
    """直前の顔の周りを切り出して拡大する `LandmarkProvider`。"""

    def __init__(
        self,
        inner: LandmarkProvider,
        *,
        policy: TrackingZoomPolicy | None = None,
    ) -> None:
        self._inner = inner
        self._policy = policy or TrackingZoomPolicy()
        self._last_bbox: BoundingBox | None = None
        self._last_full_frame_check_ms: int | None = None

    def detect_video(
        self,
        rgb_image: NDArray[np.uint8],
        *,
        frame_id: int,
        timestamp_ms: int,
    ) -> LandmarkFrame:
        height, width = rgb_image.shape[:2]
        window = self._select_window(timestamp_ms)
        crop_image, crop_box = self._crop(rgb_image, window)
        detected = self._inner.detect_video(
            crop_image, frame_id=frame_id, timestamp_ms=timestamp_ms
        )

        if not detected.has_single_face:
            self._last_bbox = None
            return build_landmark_frame(
                frame_id=detected.frame_id,
                face_count=detected.face_count,
                width=width,
                height=height,
            )

        remapped = remap_points(detected.points, crop_box)
        result = build_landmark_frame(
            frame_id=detected.frame_id,
            face_count=1,
            points=remapped,
            pose=detected.pose,
            width=width,
            height=height,
            # 奥行きは幅比だけで戻す（§8.1.1）。切り出し窓の高さも拡大率も効かない。
            depths=remap_depths(detected.depths, crop_box),
        )
        self._last_bbox = bounding_box_of(remapped)
        return result

    def close(self) -> None:
        self._inner.close()

    @property
    def policy(self) -> TrackingZoomPolicy:
        """現在有効なズーム方針（§15.2、§14.3c の自動調整の読み戻し）。"""
        return self._policy

    def update_policy(self, policy: TrackingZoomPolicy) -> None:
        """ズーム方針を差し替える（§15.2 の Advanced 設定）。

        直前に追跡した bbox は保持する。方針が変わったことは人が動いたことを
        意味しないので、全フレーム再探索を強制せず、新しい余白と窓をその bbox
        へ直ちに適用する。
        """
        self._policy = policy

    def _select_window(self, timestamp_ms: int) -> BoundingBox:
        if self._last_bbox is None or self._due_for_full_frame_recheck(timestamp_ms):
            self._last_full_frame_check_ms = timestamp_ms
            return FULL_FRAME
        return next_zoom_window(self._last_bbox, policy=self._policy)

    def _due_for_full_frame_recheck(self, timestamp_ms: int) -> bool:
        if self._last_full_frame_check_ms is None:
            return True
        elapsed = timestamp_ms - self._last_full_frame_check_ms
        return elapsed >= self._policy.recheck_interval_ms

    def _crop(
        self,
        rgb_image: NDArray[np.uint8],
        window: BoundingBox,
    ) -> tuple[NDArray[np.uint8], BoundingBox]:
        height, width = rgb_image.shape[:2]
        x0 = round(window.x_min * width)
        y0 = round(window.y_min * height)
        x1 = min(width, max(x0 + 1, round(window.x_max * width)))
        y1 = min(height, max(y0 + 1, round(window.y_max * height)))
        cropped: NDArray[np.uint8] = np.ascontiguousarray(rgb_image[y0:y1, x0:x1])

        scale = resize_scale_for(cropped.shape[1], cropped.shape[0], policy=self._policy)
        if scale > 1.0:
            new_size = (
                max(1, round(cropped.shape[1] * scale)),
                max(1, round(cropped.shape[0] * scale)),
            )
            cropped = np.asarray(
                Image.fromarray(cropped).resize(new_size, resample=Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )

        actual_box = BoundingBox(x0 / width, y0 / height, x1 / width, y1 / height)
        return cropped, actual_box
