"""Delaunay + piecewise affine warping with OpenCV (§8.6.4, FR-035).

Implements `application.ports.PiecewiseWarper`. This is the only part of the
symmetrized-image feature that needs a pixel library, which is why it is an
adapter: `domain.warp_mesh` already decided which points move, which are
pinned, and whether the displacement is safe at all.

`cv2` comes from `opencv-contrib-python` and nothing else (invariant 9). No
model is loaded and no file is written: the warp is arithmetic over an array
the caller owns.

Method, per §8.6.4: triangulate the *target* points so the output tiles
without gaps or overlaps, then reuse the same index triples on the source and
transfer each triangle with its own affine map. Triangles the mesh cannot
support (degenerate, or with vertices that do not resolve back to an index)
are skipped, leaving the original pixels in place rather than guessing.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.warp_mesh import WarpMesh, is_degenerate_triangle


class PiecewiseWarpError(ValueError):
    """The warp could not be performed on this input."""


#: Vertices are matched back to point indices through a rounded-coordinate
#: lookup, because `cv2.Subdiv2D` returns coordinates rather than the indices
#: it was given. Rounding to this many decimals absorbs the float32 round-trip
#: inside Subdiv2D without merging genuinely distinct landmarks.
_VERTEX_DECIMALS = 2


class OpenCvPiecewiseWarper:
    """Warp a face image onto its symmetrized landmark positions."""

    def warp(self, rgb_image: NDArray[np.uint8], mesh: WarpMesh) -> NDArray[np.uint8]:
        """Return a copy of ``rgb_image`` deformed source→target by ``mesh``.

        The returned image starts as a copy of the original, so any region no
        triangle covers keeps its original pixels instead of going black.
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PiecewiseWarpError("ワーピングには 3 チャンネルの RGB 画像が必要です。")
        height, width = rgb_image.shape[:2]
        if (width, height) != (mesh.width, mesh.height):
            raise PiecewiseWarpError("画像サイズが変形メッシュと一致しません。")

        warped = rgb_image.copy()
        for triangle in self._triangles(mesh):
            source_triangle = mesh.source[list(triangle)]
            target_triangle = mesh.target[list(triangle)]
            if is_degenerate_triangle(source_triangle) or is_degenerate_triangle(target_triangle):
                continue
            _transfer_triangle(rgb_image, warped, source_triangle, target_triangle)
        return warped

    def _triangles(self, mesh: WarpMesh) -> tuple[tuple[int, int, int], ...]:
        """Delaunay triangles over the target points, as index triples."""
        # The rectangle is padded by one pixel: Subdiv2D treats its bounds as
        # half-open and rejects a point sitting exactly on the far edge, which
        # is precisely where `build_warp_mesh` clamps the border anchors to.
        subdiv = cv2.Subdiv2D((0, 0, mesh.width + 1, mesh.height + 1))
        lookup: dict[tuple[float, float], int] = {}
        for index, point in enumerate(mesh.target):
            key = _vertex_key(point)
            if key in lookup:
                # A duplicate coordinate cannot be told apart afterwards, so
                # the first index wins and triangles resolving to it simply
                # move both points together.
                continue
            lookup[key] = index
            try:
                subdiv.insert((float(point[0]), float(point[1])))
            except cv2.error as error:  # pragma: no cover - guarded by clamping
                raise PiecewiseWarpError(f"三角形分割に失敗しました: {error}") from error

        triangles: list[tuple[int, int, int]] = []
        for entry in subdiv.getTriangleList():
            corners = (
                _vertex_key(entry[0:2]),
                _vertex_key(entry[2:4]),
                _vertex_key(entry[4:6]),
            )
            resolved = tuple(lookup[corner] for corner in corners if corner in lookup)
            # Subdiv2D also reports triangles touching its virtual outer
            # vertices; those have corners that were never inserted.
            if len(resolved) == 3:
                triangles.append((resolved[0], resolved[1], resolved[2]))
        return tuple(triangles)


def _vertex_key(point: Sequence[float] | NDArray[np.float64]) -> tuple[float, float]:
    return (round(float(point[0]), _VERTEX_DECIMALS), round(float(point[1]), _VERTEX_DECIMALS))


def _transfer_triangle(
    source_image: NDArray[np.uint8],
    destination: NDArray[np.uint8],
    source_triangle: NDArray[np.float64],
    target_triangle: NDArray[np.float64],
) -> None:
    """Affine-map one triangle's pixels into ``destination`` in place.

    Works on the two bounding boxes rather than the whole frame, so cost
    scales with the triangles' size instead of the image's, and blends with a
    filled mask so only the triangle itself is written.
    """
    source_box = cv2.boundingRect(np.asarray([source_triangle], dtype=np.float32))
    target_box = cv2.boundingRect(np.asarray([target_triangle], dtype=np.float32))
    if source_box[2] <= 0 or source_box[3] <= 0 or target_box[2] <= 0 or target_box[3] <= 0:
        return

    source_local = np.asarray(
        [[point[0] - source_box[0], point[1] - source_box[1]] for point in source_triangle],
        dtype=np.float32,
    )
    target_local = np.asarray(
        [[point[0] - target_box[0], point[1] - target_box[1]] for point in target_triangle],
        dtype=np.float32,
    )
    patch = source_image[
        source_box[1] : source_box[1] + source_box[3],
        source_box[0] : source_box[0] + source_box[2],
    ]
    if patch.size == 0:
        return

    affine = cv2.getAffineTransform(source_local, target_local)
    warped_patch = cv2.warpAffine(
        patch,
        affine,
        (target_box[2], target_box[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((target_box[3], target_box[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(target_local, dtype=np.int32), (1, 1, 1), cv2.LINE_AA)
    region = destination[
        target_box[1] : target_box[1] + target_box[3],
        target_box[0] : target_box[0] + target_box[2],
    ]
    if region.shape != warped_patch.shape:  # pragma: no cover - clipped at the border
        return
    np.copyto(region, warped_patch, where=mask.astype(bool))
