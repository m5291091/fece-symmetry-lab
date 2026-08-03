"""Pure geometry for the symmetrized-image warp (§8.6.4, FR-035).

Everything here is the part of piecewise warping that can be reasoned about
without touching a pixel: which points take part, which stay pinned, and when
the whole operation is too distorted to be worth performing. The Delaunay
triangulation and the per-triangle pixel transfer need `cv2` and therefore
live in `adapters.imaging` behind `application.ports.PiecewiseWarper`.

The mesh pins the image border. Landmarks cover the face but not the ears,
hair, neck, or background, so without fixed anchors those regions would either
fall outside the triangulation entirely (leaving holes) or be dragged by
whichever face triangle happened to reach them. Corners plus edge midpoints
give the triangulation a frame to tile against, and everything between an
anchor and a moved landmark interpolates smoothly.

Nothing here is a calibrated metric parameter. `MAX_WARP_DISPLACEMENT_RATIO`
is a breakdown guard in the same family as the brightness clamp: it decides
when to refuse, not what a face scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: §8.6.4: refuse the warp when any point would travel further than this
#: fraction of the interocular distance. Real asymmetry sits far below it
#: (§8.6 distances are typically a few percent), so this only ever fires on
#: a detection breakdown — where warping would produce an image that is not
#: a person rather than a symmetrized one.
MAX_WARP_DISPLACEMENT_RATIO: Final[float] = 0.25

#: A triangle thinner than this (in px²) cannot define a stable affine map.
MIN_TRIANGLE_AREA_PX: Final[float] = 1e-3

#: Corners plus edge midpoints (§8.6.4).
BORDER_ANCHOR_COUNT: Final[int] = 8


class WarpMeshError(ValueError):
    """The requested warp cannot be built or would not be safe to perform."""


@dataclass(frozen=True, slots=True)
class WarpMesh:
    """Matched source/target point sets, border anchors included.

    `source[i]` is warped to `target[i]`. The last `BORDER_ANCHOR_COUNT`
    entries are the pinned image border, identical in both, so a consumer
    triangulating `target` covers the whole frame.
    """

    source: NDArray[np.float64]
    target: NDArray[np.float64]
    width: int
    height: int
    #: How many leading entries came from landmarks rather than the border.
    landmark_count: int

    @property
    def anchor_indices(self) -> tuple[int, ...]:
        """Indices of the pinned border points."""
        return tuple(range(self.landmark_count, len(self.source)))

    @property
    def max_displacement_px(self) -> float:
        """Largest distance any point travels, in pixels."""
        if len(self.source) == 0:
            return 0.0
        return float(np.max(np.linalg.norm(self.target - self.source, axis=1)))


def border_anchors(width: int, height: int) -> NDArray[np.float64]:
    """The four corners and four edge midpoints of an image.

    Returned in source coordinates; the caller uses the same array for the
    target, which is what pins them (§8.6.4).
    """
    if width <= 0 or height <= 0:
        raise WarpMeshError("画像サイズは正の値である必要があります。")
    right = float(width - 1)
    bottom = float(height - 1)
    middle_x = right / 2.0
    middle_y = bottom / 2.0
    return np.array(
        [
            [0.0, 0.0],
            [middle_x, 0.0],
            [right, 0.0],
            [right, middle_y],
            [right, bottom],
            [middle_x, bottom],
            [0.0, bottom],
            [0.0, middle_y],
        ],
        dtype=np.float64,
    )


def triangle_area(points: NDArray[np.float64]) -> float:
    """Absolute area of one triangle given as a (3, 2) array."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (3, 2):
        raise WarpMeshError("三角形は (3, 2) の配列である必要があります。")
    first = array[1] - array[0]
    second = array[2] - array[0]
    return float(abs(first[0] * second[1] - first[1] * second[0]) / 2.0)


def is_degenerate_triangle(points: NDArray[np.float64]) -> bool:
    """Whether a triangle is too thin to define a stable affine transform."""
    return triangle_area(points) < MIN_TRIANGLE_AREA_PX


def build_warp_mesh(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    width: int,
    height: int,
    interocular_px: float,
    max_displacement_ratio: float = MAX_WARP_DISPLACEMENT_RATIO,
) -> WarpMesh:
    """Pair up the measured and symmetrized points and pin the image border.

    Both sets are clamped into the frame, because a triangulation cannot
    accept a vertex outside its bounding rectangle and a landmark sitting a
    pixel outside the edge is a detection artefact, not a reason to fail.

    Raises `WarpMeshError` when the displacement guard trips: past that point
    the piecewise map folds over itself and the output stops being a face at
    all, so refusing is the honest answer rather than showing the wreckage.
    """
    source_array = _as_points(source, "変形元")
    target_array = _as_points(target, "変形先")
    if source_array.shape != target_array.shape:
        raise WarpMeshError("変形元と変形先の点数が一致しません。")
    if len(source_array) == 0:
        raise WarpMeshError("変形にはランドマークが必要です。")
    if width <= 0 or height <= 0:
        raise WarpMeshError("画像サイズは正の値である必要があります。")
    if not np.isfinite(interocular_px) or interocular_px <= 0:
        raise WarpMeshError("両眼間隔は正の有限値である必要があります。")
    if not np.isfinite(max_displacement_ratio) or max_displacement_ratio <= 0:
        raise WarpMeshError("変位の上限比は正の有限値である必要があります。")

    displacement = float(np.max(np.linalg.norm(target_array - source_array, axis=1)))
    limit = max_displacement_ratio * interocular_px
    if displacement > limit:
        raise WarpMeshError(
            "変形量が大きすぎるため画像を生成しません"
            f"（最大 {displacement:.1f}px > 上限 {limit:.1f}px）。"
        )

    clamped_source = _clamp(source_array, width, height)
    clamped_target = _clamp(target_array, width, height)
    anchors = border_anchors(width, height)
    return WarpMesh(
        source=np.vstack([clamped_source, anchors]),
        target=np.vstack([clamped_target, anchors]),
        width=width,
        height=height,
        landmark_count=len(clamped_source),
    )


def _as_points(points: NDArray[np.float64], label: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise WarpMeshError(f"{label}の座標は (N, 2) の配列である必要があります。")
    if not np.isfinite(array).all():
        raise WarpMeshError(f"{label}の座標は有限値である必要があります。")
    return array


def _clamp(points: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]:
    clamped = points.copy()
    np.clip(clamped[:, 0], 0.0, float(width - 1), out=clamped[:, 0])
    np.clip(clamped[:, 1], 0.0, float(height - 1), out=clamped[:, 1])
    return clamped
