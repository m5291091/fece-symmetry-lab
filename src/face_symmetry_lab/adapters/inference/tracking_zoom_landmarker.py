"""Wraps a video `LandmarkProvider` with tracking-zoom crop/enlarge (§14.3a).

A PC webcam frames a face far smaller than a phone selfie does. This adapter
crops around the previously detected face, enlarges the crop, and hands that
to the wrapped provider instead of the raw frame, so the detector gets more
effective resolution on the face regardless of how far the user sits. It
never changes what the wrapped provider does; it only changes what image it
sees and re-projects the result back into full-frame coordinates, so
downstream quality gates, scoring, and overlays are unaffected.

It periodically forces a full-frame detection (`recheck_interval_ms`) even
while locked onto a face, because a face outside the current crop would
otherwise never be seen — silently keeping a second face invisible would
violate the single-face policy (FR-008).
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
    remap_points,
    resize_scale_for,
)


class TrackingZoomLandmarker:
    """A `LandmarkProvider` that crops and enlarges around the last face."""

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
        )
        self._last_bbox = bounding_box_of(remapped)
        return result

    def close(self) -> None:
        self._inner.close()

    @property
    def policy(self) -> TrackingZoomPolicy:
        """The currently active zoom policy (§15.2, §14.3c auto-tuning readback)."""
        return self._policy

    def update_policy(self, policy: TrackingZoomPolicy) -> None:
        """Swap the zoom policy (§15.2 Advanced settings).

        Keeps the last tracked bbox: a policy change does not mean the
        person moved, so the new margin/window applies to it immediately
        instead of forcing a full-frame re-search.
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
