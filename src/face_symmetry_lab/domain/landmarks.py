"""Immutable landmark DTOs and the pure conversions the overlay needs.

This module holds tracking data only. It deliberately does not express
`analysis_status`, symmetry scores, or quality gates: landmark presence alone
says nothing about whether a frame is fit for measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

#: Face Landmarker returns 478 points; the frame contract does not require it,
#: so downstream code can also work with reduced fixtures.
FACE_LANDMARK_COUNT = 478


class LandmarkError(ValueError):
    """Invalid landmark data at the domain boundary."""


class TrackingStatus(StrEnum):
    """How many faces a frame carried. Not the analysis status of §6.2."""

    NO_FACE = "NO_FACE"
    SINGLE_FACE = "SINGLE_FACE"
    MULTI_FACE = "MULTI_FACE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    """One detection result, tied to the frame it was computed from."""

    frame_id: int
    face_count: int
    points: tuple[tuple[float, float], ...] = ()

    @property
    def has_single_face(self) -> bool:
        return self.face_count == 1

    @property
    def status(self) -> TrackingStatus:
        if self.face_count == 0:
            return TrackingStatus.NO_FACE
        if self.face_count == 1:
            return TrackingStatus.SINGLE_FACE
        return TrackingStatus.MULTI_FACE


def build_landmark_frame(
    *,
    frame_id: int,
    face_count: int,
    points: tuple[tuple[float, float], ...] = (),
) -> LandmarkFrame:
    """Validate a detection result before it can reach the overlay or the UI."""
    if frame_id < 0:
        raise LandmarkError("frame_id は 0 以上である必要があります。")
    if face_count < 0:
        raise LandmarkError("face_count は 0 以上である必要があります。")
    if face_count != 1 and points:
        raise LandmarkError("単一顔のフレームだけがランドマークを保持できます。")
    for x, y in points:
        if not (isfinite(x) and isfinite(y)):
            raise LandmarkError("ランドマーク座標は有限値である必要があります。")
    return LandmarkFrame(frame_id=frame_id, face_count=face_count, points=points)


def to_pixel_points(
    frame: LandmarkFrame,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    """Convert normalized landmarks to pixel coordinates, clamped to the image."""
    if width <= 0 or height <= 0:
        raise LandmarkError("画像サイズは正の値である必要があります。")
    if not frame.has_single_face:
        return ()
    max_x = width - 1
    max_y = height - 1
    return tuple(
        (
            min(max(int(x * width), 0), max_x),
            min(max(int(y * height), 0), max_y),
        )
        for x, y in frame.points
    )
