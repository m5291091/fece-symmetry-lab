"""Pure NumPy landmark painting.

Drawing lives here rather than in an adapter because it is arithmetic on an
array: no OpenCV, MediaPipe, or Streamlit import is involved.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.geometry import GeometryError, fit_midline
from face_symmetry_lab.domain.landmarks import (
    LandmarkError,
    LandmarkFrame,
    to_pixel_array,
    to_pixel_points,
)

DEFAULT_POINT_COLOR: tuple[int, int, int] = (0, 220, 255)
DEFAULT_POINT_RADIUS = 1
DEFAULT_MIDLINE_COLOR: tuple[int, int, int] = (236, 72, 153)
DEFAULT_MIDLINE_THICKNESS = 5


def draw_landmark_points(
    rgb_image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    color: tuple[int, int, int] = DEFAULT_POINT_COLOR,
    radius: int = DEFAULT_POINT_RADIUS,
) -> NDArray[np.uint8]:
    """Return a copy of ``rgb_image`` with the frame's landmarks painted on it."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise LandmarkError("オーバーレイには 3 チャンネルの RGB 画像が必要です。")
    if radius < 0:
        raise LandmarkError("radius は 0 以上である必要があります。")

    height, width = rgb_image.shape[:2]
    pixels = to_pixel_points(frame, width=width, height=height)
    painted = rgb_image.copy()
    if not pixels:
        return painted

    xs = np.fromiter((x for x, _ in pixels), dtype=np.int64, count=len(pixels))
    ys = np.fromiter((y for _, y in pixels), dtype=np.int64, count=len(pixels))
    paint = np.asarray(color, dtype=painted.dtype)
    for offset_y in range(-radius, radius + 1):
        rows = np.clip(ys + offset_y, 0, height - 1)
        for offset_x in range(-radius, radius + 1):
            columns = np.clip(xs + offset_x, 0, width - 1)
            painted[rows, columns] = paint
    return painted


def draw_face_midline(
    rgb_image: NDArray[np.uint8],
    frame: LandmarkFrame,
    *,
    midline_indices: tuple[int, ...],
    color: tuple[int, int, int] = DEFAULT_MIDLINE_COLOR,
    thickness: int = DEFAULT_MIDLINE_THICKNESS,
) -> NDArray[np.uint8]:
    """Draw the TLS midline only across the detected face extent."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise LandmarkError("オーバーレイには 3 チャンネルの RGB 画像が必要です。")
    if thickness < 1:
        raise LandmarkError("thickness は 1 以上である必要があります。")
    painted = rgb_image.copy()
    if not frame.has_single_face or not frame.points or len(midline_indices) < 3:
        return painted

    try:
        points = to_pixel_array(frame)
        midline = fit_midline(points, midline_indices)
    except (GeometryError, LandmarkError):
        return painted

    parameters = (points - midline.origin) @ midline.direction
    start = midline.origin + float(parameters.min()) * midline.direction
    end = midline.origin + float(parameters.max()) * midline.direction
    steps = max(1, round(float(np.max(np.abs(end - start))))) + 1
    xs = np.rint(np.linspace(start[0], end[0], steps)).astype(np.int64)
    ys = np.rint(np.linspace(start[1], end[1], steps)).astype(np.int64)
    height, width = painted.shape[:2]
    paint = np.asarray(color, dtype=painted.dtype)
    negative_extent = (thickness - 1) // 2
    positive_extent = thickness // 2
    for offset_y in range(-negative_extent, positive_extent + 1):
        rows = np.clip(ys + offset_y, 0, height - 1)
        for offset_x in range(-negative_extent, positive_extent + 1):
            columns = np.clip(xs + offset_x, 0, width - 1)
            painted[rows, columns] = paint
    return painted
